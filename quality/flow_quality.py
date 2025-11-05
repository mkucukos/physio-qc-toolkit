import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, welch, correlate
import neurokit2 as nk
import json
from datetime import datetime, timezone, timedelta

# ---- helper metrics ----
def check_clipping(signal, digital_min=-100, digital_max=100, edge_pct=0.01):
    if signal.size == 0:
        return np.nan
    lower, upper = digital_min * (1 - edge_pct), digital_max * (1 - edge_pct)
    clipped = (signal <= lower) | (signal >= upper)
    return float(np.mean(clipped))

def flatline_ratio(signal, eps=1e-6):
    """
    Compute a robust flatline ratio for a 1D signal.
    Returns a single float (0–1), same as the original version.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2:
        return 1.0

    # --- Robust flatline / dropped-signal detection ---
    epoch_var = np.var(signal)
    epoch_ptp = np.ptp(signal)

    # also look for long runs of identical samples (true flat signal)
    diffs = np.diff(signal)
    repeat_ratio = np.mean(np.abs(diffs) < eps)

    # dynamic thresholds
    var_thresh = np.percentile(epoch_var, 5) * 0.2
    amp_thresh = np.percentile(epoch_ptp, 5) * 0.2

    flat_mask = ((epoch_var < var_thresh) & (epoch_ptp < amp_thresh)) or (repeat_ratio > 0.98)

    return float(flat_mask)


def missing_ratio(n_present, n_expected):
    if n_expected <= 0:
        return np.nan
    return float(max(0.0, 1.0 - n_present / n_expected))

def bandpass_filter(sig, fs, lo=0.10, hi=1.00, order=4):
    nyq = 0.5 * fs
    lo_n, hi_n = max(lo / nyq, 1e-6), min(hi / nyq, 0.999999)
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, sig, method="gust")

def bpm_welch(seg, fs, band=(0.10, 1.00)):
    if seg.size == 0:
        return np.nan
    f, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 2048))
    m = (f >= band[0]) & (f <= band[1])
    if np.any(m) and np.nansum(pxx[m]) > 0:
        dom = f[m][np.argmax(pxx[m])]
        return float(dom * 60.0)
    return np.nan

# ---- autocorrelation quality ----
def autocorr_quality(seg, fs, max_lag_sec=10,
                     digital_min=-100, digital_max=100, clip_thresh=0.01):
    """Compute normalized autocorrelation peak within lag window.
    Returns 0 for flatline or heavily clipped segments.
    """
    if len(seg) < fs:
        return np.nan

    # --- Flatline detection ---
    if np.nanstd(seg) < 1e-6:  # nearly constant signal
        return 0.0  # ✅ flatline → poor autocorr

    # --- Clipping detection ---
    lower, upper = digital_min * (1 - clip_thresh), digital_max * (1 - clip_thresh)
    clipped = (seg <= lower) | (seg >= upper)
    if np.mean(clipped) >= clip_thresh:
        return 0.0  # ✅ heavily clipped → poor autocorr

    # --- Autocorrelation computation ---
    seg = (seg - np.nanmean(seg)) / (np.nanstd(seg) + 1e-8)
    ac = correlate(seg, seg, mode='full')
    ac = ac[len(ac)//2:]  # keep positive lags

    if np.nanmax(np.abs(ac)) == 0:
        return np.nan

    ac /= np.nanmax(ac)
    lags = np.arange(len(ac)) / fs
    mask = (lags >= 1.0) & (lags <= max_lag_sec)

    return float(np.nanmax(ac[mask])) if np.any(mask) else np.nan

def ratio_summary(bad_n, total):
    good_n = total - bad_n
    return {
        "good_epochs": int(good_n),
        "bad_epochs": int(bad_n),
        "good_ratio": round(good_n / total, 3) if total else None,
        "bad_ratio": round(bad_n / total, 3) if total else None,
    }

# ---- main QC ----
def run_flow_qc(
    channel_name,
    channel_dataframes,
    fs=100,
    epoch_len=30,
    json_path=None,
    plot="per-metric",
    clipping_max=0.50,
    flatline_max=0.50,
    missing_max=0.50,
    bpm_min=10.0,
    bpm_max=22.0,
    auto_min=0.5   # 🔸 threshold for autocorrelation quality
):
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found.")

    df = channel_dataframes[channel_name]
    t_abs = pd.to_datetime(df["Absolute Time"], errors="coerce")
    if getattr(t_abs.dt, "tz", None) is None:
        t_abs = t_abs.dt.tz_localize("UTC")

    t_abs_ns = t_abs.astype("int64", copy=False)
    sig_np = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(sig_np) & np.isfinite(t_abs_ns)
    sig, t_abs_ns = sig_np[mask].astype(np.float32), t_abs_ns[mask]

    if sig.size == 0:
        empty = {
            "total_epochs": 0, "good_epochs": 0, "bad_epochs": 0,
            "good_ratio": None, "bad_ratio": None
        }
        if json_path:
            with open(json_path, "w") as f:
                json.dump({"per_epoch": [], "per_metric": {}, "overall": empty}, f, indent=2)
        return [], {}, empty

    # --- prepare time base ---
    t0_ns = t_abs_ns[0]
    t_sec = (t_abs_ns - t0_ns) / 1e9
    t0_dt = datetime.fromtimestamp(t0_ns / 1e9, tz=timezone.utc)

    spp = int(fs * epoch_len)
    starts = np.arange(0, len(sig), spp, dtype=int)
    ends = np.minimum(starts + spp, len(sig))

    per_epoch = []
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        seg = sig[s:e]
        clip = check_clipping(seg)
        flat = flatline_ratio(seg)
        miss = missing_ratio(seg.size, spp)

        try:
            seg_filt = bandpass_filter(seg - np.nanmedian(seg), fs)
        except Exception:
            seg_filt = seg

        bpm = np.nan
        try:
            rr = nk.rsp_rate(seg_filt, sampling_rate=fs, method="fft")
            if rr is not None and np.size(rr):
                bpm = float(np.nanmedian(rr))
            if not np.isfinite(bpm):
                bpm2 = nk.rsp_rate(seg_filt, sampling_rate=fs, method="count")
                if bpm2 is not None and np.size(bpm2):
                    bpm = float(np.nanmedian(bpm2))
        except Exception:
            pass
        if not np.isfinite(bpm):
            bpm = bpm_welch(seg_filt, fs)

        # --- Autocorrelation ---
        ac_score = autocorr_quality(seg_filt, fs)
        bad_auto = bool(np.isfinite(ac_score) and ac_score < auto_min)

        # --- QC flags ---
        bad_clip = bool(np.isfinite(clip) and clip > clipping_max)
        bad_flat = bool(np.isfinite(flat) and flat > flatline_max)
        bad_miss = bool(np.isfinite(miss) and miss > missing_max)
        bad_bpm_nan = bool(not np.isfinite(bpm))
        bad_bpm_low = bool(np.isfinite(bpm) and bpm < bpm_min)
        bad_bpm_high = bool(np.isfinite(bpm) and bpm > bpm_max)
        bad_bpm = bool(bad_bpm_low or bad_bpm_high or bad_bpm_nan)
        bad_epoch = bool(bad_clip or bad_flat or bad_miss or bad_bpm or bad_auto)

        per_epoch.append({
            "Epoch": int(i),
            "Start_Time_ISO": (t0_dt + timedelta(seconds=float(t_sec[s]))).isoformat(),
            "End_Time_ISO": (t0_dt + timedelta(seconds=float(t_sec[e - 1]))).isoformat(),
            "Clipping_Ratio": float(clip) if np.isfinite(clip) else None,
            "Flatline_Ratio": float(flat) if np.isfinite(flat) else None,
            "Missing_Ratio": float(miss) if np.isfinite(miss) else None,
            "BPM": float(bpm) if np.isfinite(bpm) else None,
            "AutoCorr_Q": float(ac_score) if np.isfinite(ac_score) else None,
            "Bad_Epoch": bad_epoch,
            "Bad_Clip": bad_clip,
            "Bad_Flatline": bad_flat,
            "Bad_Missing": bad_miss,
            "Bad_BPM": bad_bpm,
            "Bad_AutoCorr": bad_auto,
            "Raw_Data": seg.tolist()
        })

    # --- summaries ---
    total = len(per_epoch)
    def count(flag): return sum(1 for r in per_epoch if r[flag])
    per_metric_json = {
        "Clipping": ratio_summary(count("Bad_Clip"), total),
        "Flatline": ratio_summary(count("Bad_Flatline"), total),
        "Missing": ratio_summary(count("Bad_Missing"), total),
        "BPM": ratio_summary(count("Bad_BPM"), total),
        "AutoCorr_Q": ratio_summary(count("Bad_AutoCorr"), total)
    }
    overall_bad = count("Bad_Epoch")
    overall_json = {
        "total_epochs": total,
        "good_epochs": total - overall_bad,
        "bad_epochs": overall_bad,
        "good_ratio": round((total - overall_bad) / total, 3) if total else None,
        "bad_ratio": round(overall_bad / total, 3) if total else None,
    }

    # --- optional JSON save ---
    if json_path:
        with open(json_path, "w") as f:
            json.dump(
                {"per_epoch": per_epoch, "per_metric": per_metric_json, "overall": overall_json},
                f, indent=2
            )

    # --- plotting ---
    if plot in ("overall", "per-metric", "both"):
        times = [t0_dt + timedelta(seconds=float(s)) for s in t_sec]

        def shade(ax, flag_key):
            for r in per_epoch:
                st, et = datetime.fromisoformat(r["Start_Time_ISO"]), datetime.fromisoformat(r["End_Time_ISO"])
                ax.axvspan(st, et, color=("red" if r[flag_key] else "green"), alpha=0.18)

        if plot in ("overall", "both"):
            fig, ax = plt.subplots(figsize=(14, 5))
            step = max(1, len(sig) // 20000)
            ax.plot(times[::step], sig[::step], lw=0.8, color="black")
            shade(ax, "Bad_Epoch")
            ax.set_title(f"{channel_name} — Overall Flow QC (Red=Bad, Green=Good)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True); plt.tight_layout(); plt.show()

        if plot in ("per-metric", "both"):
            flag_map = {
                "Clipping": "Bad_Clip",
                "Flatline": "Bad_Flatline",
                "Missing": "Bad_Missing",
                "BPM": "Bad_BPM",
                "AutoCorr_Q": "Bad_AutoCorr"
            }
            for metric, flag in flag_map.items():
                fig, ax = plt.subplots(figsize=(14, 5))
                step = max(1, len(sig) // 20000)
                ax.plot(times[::step], sig[::step], lw=0.8, color="black")
                shade(ax, flag)
                ax.set_title(f"{channel_name} — {metric} QC (Red=Bad, Green=Good)")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax.grid(True); plt.tight_layout(); plt.show()

    return per_epoch, per_metric_json, overall_json
