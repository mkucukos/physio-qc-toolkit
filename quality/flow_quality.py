import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, welch
import neurokit2 as nk

# ---- helper metrics ----
def check_clipping(signal, digital_min=-100, digital_max=100, edge_pct=0.01):
    if len(signal) == 0:
        return np.nan
    lower = digital_min * (1 - edge_pct)
    upper = digital_max * (1 - edge_pct)
    clipped = np.logical_or(signal <= lower, signal >= upper)
    return float(np.mean(clipped))

def flatline_ratio(signal, eps=1e-6):
    if len(signal) < 2:
        return np.nan
    diffs = np.abs(np.diff(signal))
    return float(np.mean(diffs < eps))

def missing_ratio(n_samples_epoch, fs, epoch_len):
    expected = int(fs * epoch_len)
    if expected <= 0:
        return np.nan
    actual = int(n_samples_epoch)
    return float(max(0.0, 1.0 - actual / expected))

def _bandpass(sig, fs, lo=0.10, hi=1.00, order=4):
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    b, a = butter(order, [lo_n, hi_n], btype="bandpass")
    return filtfilt(b, a, sig, method="gust")

def _bpm_welch(seg, fs, band=(0.10, 1.00)):
    f, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 2048))
    low, high = band
    m = (f >= low) & (f <= high) & np.isfinite(pxx)
    if np.any(m) and np.nansum(pxx[m]) > 0:
        dom = f[m][np.argmax(pxx[m])]
        return float(dom * 60.0)
    return np.nan

# ---- main QC ----
def run_flow_qc(
    channel_name,
    channel_dataframes,
    fs=100,
    epoch_len=30,
    plot='overall',
    tz="UTC",
    digital_min=-100,
    digital_max=100,
    clipping_edge_pct=0.01,
    clipping_max=0.50,
    flatline_max=0.50,
    missing_max=0.50,
    bpm_min=7.0,
    bpm_max=30.0,
    bp_lo=0.10,
    bp_hi=1.00,
):
    """
    Flow QC with bandpassed-BPM estimation (NeuroKit2 -> Welch fallback).
    Marks epochs as bad if:
      - Clipping_Ratio > 0.50
      - Flatline_Ratio > 0.50
      - Missing_Ratio > 0.50
      - BPM < 7, > 30, or NaN
    """
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found in provided channel_dataframes.")

    # --- load & sanitize ---
    df = channel_dataframes[channel_name].copy()
    df["Absolute Time"] = pd.to_datetime(df["Absolute Time"], errors="coerce")
    df = df.dropna(subset=["Absolute Time", channel_name]).sort_values("Absolute Time")
    if df["Absolute Time"].dt.tz is None or str(df["Absolute Time"].dt.tz.iloc[0]) == "None":
        df["Absolute Time"] = df["Absolute Time"].dt.tz_localize(tz)

    sig_series = pd.to_numeric(df[channel_name], errors="coerce")
    mask = ~sig_series.isna()
    df = df.loc[mask].copy()
    sig = sig_series.loc[mask].to_numpy()
    times = df["Absolute Time"].to_numpy()

    samples_per_epoch = int(fs * epoch_len)
    n_epochs = int(np.ceil(len(sig) / samples_per_epoch)) if samples_per_epoch > 0 else 0

    rows = []
    for i in range(n_epochs):
        s = i * samples_per_epoch
        e = min((i + 1) * samples_per_epoch, len(sig))
        if s >= e:
            continue
        epoch_raw = sig[s:e]

        # metrics
        clip = check_clipping(epoch_raw, digital_min=digital_min, digital_max=digital_max, edge_pct=clipping_edge_pct)
        flat = flatline_ratio(epoch_raw, eps=1e-6)
        miss = missing_ratio(len(epoch_raw), fs, epoch_len)

        # BPM estimation
        bpm = np.nan
        try:
            epoch_filt = _bandpass(epoch_raw - np.nanmedian(epoch_raw), fs, lo=bp_lo, hi=bp_hi)
        except Exception:
            epoch_filt = epoch_raw
        try:
            rr = nk.rsp_rate(epoch_filt, sampling_rate=fs, method="fft")
            if rr is not None and np.size(rr):
                bpm = float(np.nanmedian(rr))
            if not np.isfinite(bpm):
                rr2 = nk.rsp_rate(epoch_filt, sampling_rate=fs, method="count")
                if rr2 is not None and np.size(rr2):
                    bpm = float(np.nanmedian(rr2))
        except Exception:
            pass
        if not np.isfinite(bpm):
            bpm = _bpm_welch(epoch_filt, fs, band=(bp_lo, bp_hi))

        # Flags (now including NaN BPM)
        bad_clip = (not np.isnan(clip)) and (clip > clipping_max)
        bad_flat = (not np.isnan(flat)) and (flat > flatline_max)
        bad_miss = (not np.isnan(miss)) and (miss > missing_max)
        bad_bpm_nan = not np.isfinite(bpm)
        bad_bpm_low = np.isfinite(bpm) and bpm < bpm_min
        bad_bpm_high = np.isfinite(bpm) and bpm > bpm_max
        bad_bpm = bad_bpm_low or bad_bpm_high or bad_bpm_nan

        bad_epoch = bool(bad_clip or bad_flat or bad_miss or bad_bpm)

        rows.append({
            "Epoch": i + 1,
            "Start_Time": times[s],
            "End_Time": times[e - 1],
            "Clipping_Ratio": clip,
            "Flatline_Ratio": flat,
            "Missing_Ratio": miss,
            "BPM": bpm,
            "Bad_Epoch": bad_epoch,
            "Bad_Clip": bad_clip,
            "Bad_Flatline": bad_flat,
            "Bad_Missing": bad_miss,
            "Bad_BPM_Low": bad_bpm_low,
            "Bad_BPM_High": bad_bpm_high,
            "Bad_BPM_NaN": bad_bpm_nan,
            "Bad_BPM": bad_bpm,
        })

    quality_df = pd.DataFrame(rows)

    # summaries
    total = len(quality_df)
    def _summary(flag_col):
        bad_n = int(quality_df[flag_col].sum()) if total else 0
        good_n = total - bad_n
        return {
            "good_epochs": good_n,
            "bad_epochs": bad_n,
            "good_ratio": round(good_n / total, 3) if total else np.nan,
            "bad_ratio": round(bad_n / total, 3) if total else np.nan,
        }

    per_metric = {
        f"Clipping>(>{clipping_max:.2f})": _summary("Bad_Clip"),
        f"Flatline>(>{flatline_max:.2f})": _summary("Bad_Flatline"),
        f"Missing>(>{missing_max:.2f})":  _summary("Bad_Missing"),
        f"BPM<({bpm_min:g})":             _summary("Bad_BPM_Low"),
        f"BPM>({bpm_max:g})":             _summary("Bad_BPM_High"),
        "BPM_NaN":                        _summary("Bad_BPM_NaN"),
    }

    overall_bad = int(quality_df["Bad_Epoch"].sum()) if total else 0
    overall_json = {
        "total_epochs": total,
        "good_epochs": total - overall_bad,
        "bad_epochs": overall_bad,
        "good_ratio": round((total - overall_bad) / total, 3) if total else np.nan,
        "bad_ratio": round(overall_bad / total, 3) if total else np.nan,
        "limits": {
            "clipping_max": clipping_max,
            "flatline_max": flatline_max,
            "missing_max":  missing_max,
            "bpm_min": bpm_min,
            "bpm_max": bpm_max,
            "bp_band_hz": [bp_lo, bp_hi],
        }
    }

    # --- plotting ---
    def _shade(ax, flag_col):
        for _, r in quality_df.iterrows():
            ax.axvspan(r["Start_Time"], r["End_Time"],
                       color=("red" if r[flag_col] else "green"), alpha=0.18)

    if total and plot in ("overall", "both"):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df["Absolute Time"], sig, lw=0.8, color="black")
        ax.set_title(f"{channel_name} — Overall QC (Red=Bad, Green=Good)")
        ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.grid(True)
        _shade(ax, "Bad_Epoch")
        plt.tight_layout(); plt.show()

    if total and plot in ("per-metric", "both"):
        for metric, flag in {
            "Clipping_Ratio": "Bad_Clip",
            "Flatline_Ratio": "Bad_Flatline",
            "Missing_Ratio": "Bad_Missing",
            "BPM (Low/High/NaN)": "Bad_BPM",
        }.items():
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df["Absolute Time"], sig, lw=0.8, color="black")
            if "BPM" in metric:
                title = f"{channel_name} — BPM QC (Red if NaN, <{bpm_min:g}, or >{bpm_max:g})"
            else:
                title = f"{channel_name} — {metric}: QC (Red=Bad, Green=Good)"
            ax.set_title(title)
            ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.grid(True)
            _shade(ax, flag)
            plt.tight_layout(); plt.show()


    return quality_df, per_metric, overall_json