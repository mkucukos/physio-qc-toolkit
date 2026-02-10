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


def calculate_flow_quality(
    channel_name,
    channel_dataframes,
    fs=100,
    epoch_len=30,
    thresholds=None,
    plot=False,
):
    # ---------------- Thresholds ----------------
    th = {
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max": 0.50,
        "bpm_min": 10.0,
        "bpm_max": 22.0,
        "auto_min": 0.50,
    }
    if thresholds:
        th.update(thresholds)

    df = channel_dataframes[channel_name]

    # ============================================================
    # TIME + SIGNAL INGEST (SAFE)
    # ============================================================

    t_abs = pd.to_datetime(df["Absolute Time"], errors="coerce", utc=True)
    t_abs = t_abs.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")

    sig = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(sig) & np.isfinite(t_abs.astype("int64"))
    sig = sig[mask]
    t_abs = t_abs[mask]

    if sig.size == 0:
        return None

    # ============================================================
    # SETUP
    # ============================================================

    spp = int(fs * epoch_len)
    n_epochs = int(np.ceil(len(sig) / spp))

    sig = sig.astype(np.float32)
    sig_demean = sig - np.nanmedian(sig)

    try:
        sig_filt = bandpass_filter(sig_demean, fs)
    except Exception:
        sig_filt = sig_demean

    epoch_start_times = t_abs[::spp][:n_epochs]

    # ============================================================
    # STORAGE
    # ============================================================

    clip_vals = np.full(n_epochs, np.nan)
    flat_vals = np.full(n_epochs, np.nan)
    miss_vals = np.full(n_epochs, np.nan)
    bpm_vals  = np.full(n_epochs, np.nan)
    ac_vals   = np.full(n_epochs, np.nan)

    bpm_skipped = np.zeros(n_epochs, dtype=bool)
    ac_skipped  = np.zeros(n_epochs, dtype=bool)

    clip_max = th["clipping_max"]
    flat_max = th["flatline_max"]
    miss_max = th["missing_max"]
    bpm_min  = th["bpm_min"]
    bpm_max  = th["bpm_max"]
    auto_min = th["auto_min"]

    # ============================================================
    # EPOCH LOOP
    # ============================================================

    for i in range(n_epochs):
        s = i * spp
        e = min(s + spp, sig.size)

        seg = sig[s:e]
        seg_f = sig_filt[s:e]

        c = clipping_ratio(seg)
        f = flatline_ratio(seg)
        m = missing_ratio(seg.size, spp)

        clip_vals[i] = c
        flat_vals[i] = f
        miss_vals[i] = m

        # ---------- HARD GATE ----------
        if (f > flat_max) or (m > miss_max) or (c > clip_max):
            bpm_skipped[i] = True
            ac_skipped[i] = True
            continue

        # ---------- BPM ----------
        bpm = np.nan
        try:
            rr = nk.rsp_rate(seg_f, sampling_rate=fs, method="fft")
            if rr is not None and rr.size:
                bpm = float(np.nanmedian(rr))
        except Exception:
            pass

        if not np.isfinite(bpm):
            bpm = bpm_welch(seg_f, fs)

        bpm_vals[i] = bpm

        # ---------- AUTOCORR ----------
        if (f > flat_max) or (m > miss_max) or (c > clip_max):
            ac_skipped[i] = True
        else:
            ac_vals[i] = autocorr_quality(seg_f, fs=fs)

    # ============================================================
    # MASKS
    # ============================================================

    masks = {
        "clipping": clip_vals > clip_max,
        "flatline": flat_vals > flat_max,
        "missing": miss_vals > miss_max,
        "bpm": (~bpm_skipped) & (
            (~np.isfinite(bpm_vals)) |
            (bpm_vals < bpm_min) |
            (bpm_vals > bpm_max)
        ),
        "autocorr_q": (~ac_skipped) & np.isfinite(ac_vals) & (ac_vals < auto_min),
    }

    combined_mask = np.any(np.column_stack(list(masks.values())), axis=1)

    # ============================================================
    # PLOTTING (WITH WHITE FOR SKIPPED)
    # ============================================================

    if plot:
        step = max(1, sig.size // 20000)

        def shade(ax, bad_mask, skip_mask=None):
            for i in range(n_epochs):
                st = epoch_start_times[i]
                et = st + np.timedelta64(epoch_len, "s")

                if skip_mask is not None and skip_mask[i]:
                    color = "white"
                else:
                    color = "red" if bad_mask[i] else "green"

                ax.axvspan(st, et, color=color, alpha=0.25, linewidth=0)

        fig, axes = plt.subplots(
            len(masks) + 1,
            1,
            figsize=(16, 1.5 * (len(masks) + 1)),
            sharex=True
        )

        axes[0].plot(t_abs[::step], sig[::step], lw=0.8, color="black")
        shade(axes[0], combined_mask)
        axes[0].set_title(f"{channel_name} — Overall Flow QC")

        for ax, name in zip(axes[1:], masks):
            ax.plot(t_abs[::step], sig[::step], lw=0.8, color="black")

            if name == "bpm":
                shade(ax, masks[name], bpm_skipped)
            elif name == "autocorr_q":
                shade(ax, masks[name], ac_skipped)
            else:
                shade(ax, masks[name])

            ax.set_title(name.upper())


        xmin = t_abs[0]
        xmax = t_abs[-1]

        for ax in axes:
            ax.set_xlim(xmin, xmax)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.tight_layout()
        plt.show()

    # ============================================================
    # RETURN
    # ============================================================

    return {
        "metric_names": list(masks.keys()),
        "metric_values": {
            "clipping": clip_vals,
            "flatline": flat_vals,
            "missing": miss_vals,
            "bpm": bpm_vals,
            "autocorr_q": ac_vals,
        },
        "bad_masks": masks,
        "combined_mask": combined_mask,
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channel": channel_name,
        },
    }