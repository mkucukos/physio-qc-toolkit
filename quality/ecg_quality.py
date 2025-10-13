import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch, butter, filtfilt
from scipy import stats
import neurokit2 as nk

# ---------------- ECG features (HR, HRV, SNR per epoch) ----------------
def get_ecg_features(ecg, time_in_sec, fs):
    ecg = np.asarray(ecg, dtype=np.float64)

    b, a = butter(4, (0.25, 25), btype='bandpass', fs=fs)
    ecg_filt = filtfilt(b, a, ecg, axis=0)
    ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)

    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="engzeemod2012")
    r_idx = rpeaks.get('ECG_R_Peaks', np.array([], dtype=int))
    if r_idx is None or len(r_idx) < 2:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    rr_times = time_in_sec[r_idx]
    d_rr = np.diff(rr_times)
    if len(d_rr) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    hr = 60.0 / d_rr
    if hr.size > 1:
        z = np.abs(stats.zscore(hr, nan_policy='omit'))
        hr = hr[z <= 6.0]
    if hr.size == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    hr_mean = np.nanmean(hr)
    hr_min  = np.nanmin(hr)
    hr_max  = np.nanmax(hr)

    d_rr_ms = np.diff(1000.0 * d_rr)
    hrv = np.sqrt(np.nanmean(d_rr_ms**2)) if d_rr_ms.size else np.nan

    sig_pow = np.var(ecg_filt)
    noise_pow = np.var(ecg_filt - ecg_cleaned)
    snr_db = 10.0 * np.log10(sig_pow / (noise_pow + 1e-12)) if sig_pow > 0 else np.nan

    return np.array([hr_mean, hr_max, hr_min, hrv, snr_db])
# ---------------- QC runner: ANY fail => epoch BAD ----------------
def run_ecg_qc(
    channel_name,
    channel_dataframes,     # ✅ now required as input
    fs=200,
    epoch_len=30,
    thresholds=None,
    json_path=None,         # optional path to save per-metric JSON
    plot='overall'          # 'overall' | 'per-metric' | 'both' | 0
):
    """
    Performs ECG signal quality control (QC) on a specific channel DataFrame.

    Parameters
    ----------
    channel_name : str
        The name of the channel to process (key in channel_dataframes).
    channel_dataframes : dict
        Dictionary mapping channel names to DataFrames
        (from your EDF reading function).
    fs : int
        Sampling frequency (Hz).
    epoch_len : int
        Length of each epoch in seconds.
    thresholds : dict
        Optional overrides for QC thresholds.
    json_path : str, optional
        Path to save per-metric JSON summary.
    plot : str
        'overall', 'per-metric', 'both', or 0 (to disable plots).

    Returns
    -------
    quality_df : DataFrame
        Per-epoch QC metrics and flags.
    per_metric : dict
        JSON summary of per-metric good/bad ratios.
    overall_json : dict
        Summary of total good/bad epoch ratio.
    """

    # --- threshold defaults ---
    th = {
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max":  0.50,
        "baseline_max": 0.15,
        "hr_min": 25.0,
        "hr_max": 220.0,
        "snr_min": 5.0,
    }
    if thresholds:
        th.update(thresholds)

    samples_per_epoch = fs * epoch_len

    # --- helper metric functions ---
    def clipping_ratio(signal, digital_min=-8333, digital_max=8333, threshold=0.01):
        lower, upper = digital_min * (1 - threshold), digital_max * (1 - threshold)
        clipped = np.logical_or(signal <= lower, signal >= upper)
        return float(np.mean(clipped))

    def flatline_ratio(signal, eps=1e-6):
        if len(signal) < 2:
            return 1.0
        diffs = np.abs(np.diff(signal))
        return float(np.mean(diffs < eps))

    def missing_ratio(signal, fs, epoch_length):
        expected = int(fs * epoch_length)
        actual = len(signal)
        if expected <= 0:
            return 1.0
        return float(max(0.0, 1.0 - actual / expected))

    def baseline_wander_ratio(signal, fs, cutoff=0.30):
        if len(signal) < fs:
            return np.nan
        f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), 2048))
        total = float(np.sum(pxx))
        if total <= 0:
            return np.nan
        return float(np.sum(pxx[f <= cutoff]) / total)

    # --- check channel existence ---
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found in provided channel_dataframes.")

    df = channel_dataframes[channel_name].copy()
    df["Absolute Time"] = pd.to_datetime(df["Absolute Time"], errors="coerce")
    if df["Absolute Time"].dt.tz is None:
        df["Absolute Time"] = df["Absolute Time"].dt.tz_localize("UTC")

    df = df.dropna(subset=["Absolute Time", channel_name]).sort_values("Absolute Time")

    sig_series = pd.to_numeric(df[channel_name], errors="coerce")
    mask = ~sig_series.isna()
    df = df.loc[mask].copy()
    sig = sig_series.loc[mask].to_numpy()

    t0 = df["Absolute Time"].iloc[0]
    time_in_sec_full = (df["Absolute Time"] - t0).dt.total_seconds().to_numpy()

    # --- epoch loop ---
    n_epochs = int(np.ceil(len(sig) / samples_per_epoch))
    rows = []
    for i in range(n_epochs):
        s = i * samples_per_epoch
        e = min((i + 1) * samples_per_epoch, len(sig))
        if s >= e:
            continue

        epoch = sig[s:e]
        t_epoch = time_in_sec_full[s:e]

        clip = clipping_ratio(epoch)
        flat = flatline_ratio(epoch)
        miss = missing_ratio(epoch, fs, epoch_len)
        base = baseline_wander_ratio(epoch, fs, cutoff=0.30)

        hr_mean = snr_db = np.nan
        try:
            hr_mean, hr_max, hr_min, hrv, snr_db = get_ecg_features(epoch, t_epoch, fs).tolist()
        except Exception:
            pass

        bad_clip = clip > th["clipping_max"]
        bad_flat = flat > th["flatline_max"]
        bad_miss = miss > th["missing_max"]
        bad_base = (not np.isnan(base)) and (base > th["baseline_max"])
        bad_hr   = (np.isnan(hr_mean)) or (hr_mean < th["hr_min"]) or (hr_mean > th["hr_max"])
        bad_snr  = (np.isnan(snr_db))  or (snr_db < th["snr_min"])
        bad_epoch = bool(bad_clip or bad_flat or bad_miss or bad_base or bad_hr or bad_snr)

        rows.append({
            "Epoch": i + 1,
            "Start_Time": df["Absolute Time"].iloc[s],
            "End_Time": df["Absolute Time"].iloc[e - 1],
            "Clipping_Ratio": clip,
            "Flatline_Ratio": flat,
            "Missing_Ratio": miss,
            "Baseline_Wander_Ratio": base,
            "HR_Mean": hr_mean,
            "SNR_dB": snr_db,
            "Bad_Epoch": bad_epoch,
            "Bad_Clip": bad_clip,
            "Bad_Flatline": bad_flat,
            "Bad_Missing": bad_miss,
            "Bad_Baseline": bad_base,
            "Bad_HR": bad_hr,
            "Bad_SNR": bad_snr
        })

    quality_df = pd.DataFrame(rows)

    # --- per-metric ratios ---
    total = len(quality_df)
    def ratio_summary(col):
        bad_n = int(quality_df[col].sum()) if total else 0
        good_n = total - bad_n
        return {
            "good_epochs": good_n,
            "bad_epochs": bad_n,
            "good_ratio": round(good_n / total, 3) if total else np.nan,
            "bad_ratio": round(bad_n / total, 3) if total else np.nan
        }

    per_metric = {
        "Clipping":  ratio_summary("Bad_Clip"),
        "Flatline":  ratio_summary("Bad_Flatline"),
        "Missing":   ratio_summary("Bad_Missing"),
        "Baseline":  ratio_summary("Bad_Baseline"),
        "HR_Mean":   ratio_summary("Bad_HR"),
        "SNR_dB":    ratio_summary("Bad_SNR"),
    }

    overall_bad = int(quality_df["Bad_Epoch"].sum()) if total else 0
    overall_json = {
        "total_epochs": total,
        "good_epochs": total - overall_bad,
        "bad_epochs": overall_bad,
        "good_ratio": round((total - overall_bad) / total, 3) if total else np.nan,
        "bad_ratio": round(overall_bad / total, 3) if total else np.nan,
    }

    if json_path:
        with open(json_path, "w") as f:
            json.dump(per_metric, f, indent=4)

    # --- plots ---
    def _shade(ax, flag):
        for _, r in quality_df.iterrows():
            ax.axvspan(r["Start_Time"], r["End_Time"],
                       color=("red" if r[flag] else "green"), alpha=0.18)

    if plot in ("overall", "both"):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df["Absolute Time"], sig, lw=0.8, color="black")
        ax.set_title(f"{channel_name} — Overall QC (Red=Bad, Green=Good)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True)
        _shade(ax, "Bad_Epoch")
        plt.tight_layout()
        plt.show()

    if plot in ("per-metric", "both"):
        metric_flag_map = {
            "Clipping_Ratio": "Bad_Clip",
            "Flatline_Ratio": "Bad_Flatline",
            "Missing_Ratio": "Bad_Missing",
            "Baseline_Wander_Ratio": "Bad_Baseline",
            "HR_Mean": "Bad_HR",
            "SNR_dB": "Bad_SNR",
        }
        for metric, flag in metric_flag_map.items():
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df["Absolute Time"], sig, lw=0.8, color="black")
            ax.set_title(f"{channel_name} — {metric}: QC (Red=Bad, Green=Good)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True)
            _shade(ax, flag)
            plt.tight_layout()
            plt.show()

    return quality_df, per_metric, overall_json