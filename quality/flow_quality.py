import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, welch, correlate
import neurokit2 as nk
from datetime import datetime, timezone, timedelta

# ---------- helpers ----------
def clipping_ratio(sig):
    """
    Detect ECG saturation (clipping) using plateau detection.
    Works without assuming fixed ADC limits.
    Returns fraction of samples near min/max rails.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size < 10:
        return np.nan

    s_min = np.nanmin(sig)
    s_max = np.nanmax(sig)
    s_rng = s_max - s_min

    if s_rng <= 1e-12:
        return 1.0  # completely flat / dead channel

    # samples "near rail" = within 0.1% of epoch range
    eps = s_rng * 0.001

    near_min = np.mean(sig <= (s_min + eps))
    near_max = np.mean(sig >= (s_max - eps))

    return float(max(near_min, near_max))


def flatline_ratio(signal, eps=1e-6):
    """
    Returns 1.0 if flatline-like, else 0.0
    (mask-based QC expects a scalar per epoch)
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < 2:
        return 1.0

    epoch_var = np.var(sig)
    epoch_ptp = np.ptp(sig)

    diffs = np.diff(sig)
    repeat_ratio = np.mean(np.abs(diffs) < eps)

    # absolute-ish thresholds (not percentile on scalar)
    if (epoch_var < 1e-12) or (epoch_ptp < 1e-6) or (repeat_ratio > 0.98):
        return 1.0
    return 0.0


def missing_ratio(n_present, n_expected):
    if n_expected <= 0:
        return np.nan
    return float(max(0.0, 1.0 - n_present / n_expected))


def bandpass_filter(sig, fs, lo=0.10, hi=1.00, order=4):
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, sig, method="gust")


def bpm_welch(seg, fs, band=(0.10, 1.00)):
    if seg.size == 0:
        return np.nan
    f, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 2048))
    m = (f >= band[0]) & (f <= band[1])
    if np.any(m):
        p = np.nansum(pxx[m])
        if np.isfinite(p) and p > 1e-12:
            dom = f[m][np.nanargmax(pxx[m])]
            return float(dom * 60.0)
    return np.nan


def autocorr_quality(seg, fs, max_lag_sec=10, digital_min=-100, digital_max=100, clip_thresh=0.01):
    """
    Normalized autocorrelation peak within lag window.
    Returns 0.0 for flatline or heavily clipped segments.
    """
    seg = np.asarray(seg, dtype=float)
    if seg.size < fs:
        return np.nan

    if np.nanstd(seg) < 1e-6:
        return 0.0

    lower = digital_min * (1 - clip_thresh)
    upper = digital_max * (1 - clip_thresh)
    clipped = (seg <= lower) | (seg >= upper)
    if np.mean(clipped) >= clip_thresh:
        return 0.0

    segz = (seg - np.nanmean(seg)) / (np.nanstd(seg) + 1e-8)
    ac = correlate(segz, segz, mode="full")
    ac = ac[len(ac) // 2:]  # positive lags

    mx = np.nanmax(np.abs(ac))
    if not np.isfinite(mx) or mx <= 1e-12:
        return np.nan

    ac = ac / mx
    lags = np.arange(len(ac)) / fs
    m = (lags >= 1.0) & (lags <= max_lag_sec)
    return float(np.nanmax(ac[m])) if np.any(m) else np.nan


# ---------- main: ECG-style output ----------
def calculate_flow_quality(
    channel_name,
    channel_dataframes,
    fs=100,
    epoch_len=30,
    thresholds=None,
    plot="per-metric",
):

    th = {
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max": 0.50,
        "bpm_min": 10.0,
        "bpm_max": 22.0,
        "auto_min": 0.50,
        "clip_edge_frac": 0.001,   # 👈 adaptive plateau threshold
    }
    if thresholds:
        th.update(thresholds)

    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found in channel_dataframes.")

    df = channel_dataframes[channel_name]

    t_abs = pd.to_datetime(df["Absolute Time"], errors="coerce")
    if getattr(t_abs.dt, "tz", None) is None:
        t_abs = t_abs.dt.tz_localize("UTC")

    t_abs_ns = t_abs.astype("int64", copy=False).to_numpy()
    sig_np = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(sig_np) & np.isfinite(t_abs_ns)
    sig = sig_np[mask].astype(np.float32)
    t_abs_ns = t_abs_ns[mask]

    if sig.size == 0:
        return {
            "metric_names": ["clipping", "flatline", "missing", "bpm", "autocorr_q"],
            "metric_values": {k: np.array([]) for k in ["clipping","flatline","missing","bpm","autocorr_q"]},
            "bad_masks": {k: np.array([], dtype=bool) for k in ["clipping","flatline","missing","bpm","autocorr_q"]},
            "combined_mask": np.array([], dtype=bool),
            "metadata": {"fs": fs, "epoch_len": epoch_len, "channel": channel_name},
        }

    spp = int(fs * epoch_len)
    n_epochs = int(np.ceil(len(sig) / spp))

    clip_vals = np.full(n_epochs, np.nan)
    flat_vals = np.full(n_epochs, np.nan)
    miss_vals = np.full(n_epochs, np.nan)
    bpm_vals  = np.full(n_epochs, np.nan)
    ac_vals   = np.full(n_epochs, np.nan)

    t0_ns = t_abs_ns[0]
    t_sec = (t_abs_ns - t0_ns) / 1e9
    t0_dt = datetime.fromtimestamp(t0_ns / 1e9, tz=timezone.utc)

    epoch_start_times = np.empty(n_epochs, dtype="datetime64[ns]")

    for i in range(n_epochs):
        s, e = i * spp, min((i + 1) * spp, len(sig))
        seg = sig[s:e]
        epoch_start_times[i] = np.datetime64(int(t_abs_ns[s]), "ns")

        # ✅ Adaptive clipping
        clip_vals[i] = clipping_ratio(seg)
        flat_vals[i] = flatline_ratio(seg)
        miss_vals[i] = missing_ratio(seg.size, spp)

        try:
            seg_filt = bandpass_filter(seg - np.nanmedian(seg), fs)
        except:
            seg_filt = seg

        bpm = np.nan
        try:
            rr = nk.rsp_rate(seg_filt, sampling_rate=fs, method="fft")
            if rr is not None and np.size(rr):
                bpm = float(np.nanmedian(rr))
            if not np.isfinite(bpm):
                rr2 = nk.rsp_rate(seg_filt, sampling_rate=fs, method="count")
                if rr2 is not None and np.size(rr2):
                    bpm = float(np.nanmedian(rr2))
        except:
            pass
        if not np.isfinite(bpm):
            bpm = bpm_welch(seg_filt, fs)

        bpm_vals[i] = bpm

        # ✅ Autocorr no longer depends on digital rails
        ac_vals[i] = autocorr_quality(seg_filt, fs=fs)

    masks = {
        "clipping": clip_vals > th["clipping_max"],
        "flatline": flat_vals > th["flatline_max"],
        "missing": miss_vals > th["missing_max"],
        "bpm": (~np.isfinite(bpm_vals)) | (bpm_vals < th["bpm_min"]) | (bpm_vals > th["bpm_max"]),
        "autocorr_q": np.isfinite(ac_vals) & (ac_vals < th["auto_min"]),
    }

    combined_mask = np.any(np.column_stack(list(masks.values())), axis=1)

    # ---------- Plotting ----------
    if plot in ("overall", "per-metric", "both"):
        times = [t0_dt + timedelta(seconds=float(s)) for s in t_sec]
        step = max(1, len(sig) // 20000)

        def shade(ax, bad_mask):
            for i, bad in enumerate(bad_mask):
                st = epoch_start_times[i].astype("datetime64[ms]").astype(object)
                et = (epoch_start_times[i] + np.timedelta64(int(epoch_len), "s")).astype("datetime64[ms]").astype(object)
                ax.axvspan(st, et, color=("red" if bad else "green"), alpha=0.18)

        if plot in ("overall", "both"):
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(times[::step], sig[::step], lw=0.8, color="black")
            shade(ax, combined_mask)
            ax.set_title(f"{channel_name} — Overall Flow QC (Green=Good, Red=Bad)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        if plot in ("per-metric", "both"):
            for name, mask in masks.items():
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(times[::step], sig[::step], lw=0.8, color="black")
                shade(ax, mask)
                ax.set_title(f"{channel_name} — {name.upper()} QC")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax.grid(True)
                plt.tight_layout()
                plt.show()

    return {
        "metric_names": ["clipping", "flatline", "missing", "bpm", "autocorr_q"],
        "metric_values": {
            "clipping": clip_vals,
            "flatline": flat_vals,
            "missing": miss_vals,
            "bpm": bpm_vals,
            "autocorr_q": ac_vals,
        },
        "bad_masks": masks,
        "combined_mask": combined_mask,
        "metadata": {"fs": fs, "epoch_len": epoch_len, "channel": channel_name},
    }