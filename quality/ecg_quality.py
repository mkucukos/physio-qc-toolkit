import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import welch, butter, filtfilt
from scipy import stats
import neurokit2 as nk


# ---------------- ECG features (HR, HRV, SNR per epoch) ----------------
def get_ecg_features(ecg, time_in_sec, fs):
    """
    Compute ECG features from raw ECG signal.

    Parameters
    ----------
    ecg : array-like
        Raw ECG signal.
    time_in_sec : array-like
        Timestamps corresponding to each sample of the ECG signal.
    fs : float
        Sampling frequency of the ECG signal.

    Returns
    -------
    array
        Array of ECG features: [mean heart rate, maximum heart rate, minimum heart rate, heart rate variability].
    """
    try:
        ecg = np.asarray(ecg, dtype=np.float64)  # Ensure the ECG signal is a numpy array of floats
        b, a = butter(4, (0.25, 25), 'bandpass', fs=fs)
        ecg_filt = filtfilt(b, a, ecg, axis=0)
        ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)
        instant_peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="engzeemod2012")
    except Exception as e:
        raise ValueError("Error processing ECG signal: " + str(e))

    rr_times = time_in_sec[rpeaks['ECG_R_Peaks']]
    if len(rr_times) == 0:
        raise ValueError("No R-peaks detected in ECG signal.")

    d_rr = np.diff(rr_times)
    heart_rate = 60 / d_rr
    if heart_rate.size == 0:
        raise ValueError("Error computing heart rate from ECG signal.")

    valid_heart_rate = heart_rate[~np.isnan(heart_rate)]
    z_scores = np.abs(stats.zscore(valid_heart_rate))

    z_score_threshold = 5.0
    heart_rate = valid_heart_rate[z_scores <= z_score_threshold]

    hr_mean = np.nanmean(heart_rate)
    hr_min = np.nanmin(heart_rate)
    hr_max = np.nanmax(heart_rate)
    d_rr_ms = 1000 * d_rr
    d_d_rr_ms = np.diff(d_rr_ms)

    valid_d_d_rr_ms = d_d_rr_ms[~np.isnan(d_d_rr_ms)]
    z_scores = np.abs(stats.zscore(valid_d_d_rr_ms))
    d_d_rr_ms = valid_d_d_rr_ms[z_scores <= z_score_threshold]
    heart_rate_variability = np.sqrt(np.nanmean(np.square(d_d_rr_ms)))

    ecg_with_rr_intervals = []
    ecg_with_rr_intervals_cleaned = []

    for rr_interval in rr_times:
        start_time = rr_interval - 0.1  # 0.1 seconds before the RR interval
        end_time = rr_interval + 0.1  # 0.1 seconds after the RR interval
        indices = np.where((time_in_sec >= start_time) & (time_in_sec <= end_time))[0]

        indices = indices[(indices >= 0) & (indices < len(ecg))]

        if len(indices) > 0:
            ecg_with_rr_intervals.extend(ecg[indices])
            ecg_with_rr_intervals_cleaned.extend(ecg_cleaned[indices])

    ecg_with_rr_intervals = np.array(ecg_with_rr_intervals)
    ecg_with_rr_intervals_cleaned = np.array(ecg_with_rr_intervals_cleaned)

    signal_power = np.var(ecg_with_rr_intervals)
    noise_power = np.var(ecg_with_rr_intervals - ecg_with_rr_intervals_cleaned)

    snr_values = 10 * np.log10(signal_power / noise_power)

    return np.array([hr_mean, hr_max, hr_min, heart_rate_variability, snr_values])


# ---------------- QC runner: ANY fail => epoch BAD ----------------
def run_ecg_qc(
    channel_name,
    channel_dataframes,
    fs=200,
    epoch_len=30,
    thresholds=None,
    json_path=None,
    plot="overall",
):
    """
    Performs ECG signal quality control (QC) using pure NumPy (no pandas).
    Returns JSON-like objects.
    """

    # --- threshold defaults ---
    th = {
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max": 0.50,
        "baseline_max": 0.30,
        "hr_min": 25.0,
        "hr_max": 220.0,
        "snr_min": 5.0,
        "inv_ratio_max": 0.5  # added inversion threshold
    }
    if thresholds:
        th.update(thresholds)

    samples_per_epoch = fs * epoch_len

    # --- extract from input dictionary ---
    qc_epoch = channel_dataframes[channel_name]
    abs_time = np.array(qc_epoch["Absolute Time"], dtype="datetime64[ns]")
    signal = np.array(qc_epoch[channel_name], dtype=np.float64)

    mask = ~np.isnan(signal)
    abs_time = abs_time[mask]
    signal = signal[mask]

    t0 = abs_time[0]
    time_in_sec = (abs_time - t0).astype("timedelta64[ns]").astype(float) / 1e9

    # --- helper metric functions ---
    def clipping_ratio(signal, digital_min=-8333, digital_max=8333, threshold=0.01):
        lower, upper = digital_min * (1 - threshold), digital_max * (1 - threshold)
        clipped = np.logical_or(signal <= lower, signal >= upper)
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

    # --- epoch loop ---
    n_epochs = int(np.ceil(len(signal) / samples_per_epoch))
    results = []

    for i in range(n_epochs):
        s = i * samples_per_epoch
        e = min((i + 1) * samples_per_epoch, len(signal))
        if s >= e:
            continue

        epoch = signal[s:e]
        t_epoch = time_in_sec[s:e]

        clip = clipping_ratio(epoch)
        flat = flatline_ratio(epoch)
        miss = missing_ratio(epoch, fs, epoch_len)
        base = baseline_wander_ratio(epoch, fs)

        hr_mean = snr_db = np.nan
        inv_ratio = np.nan
        was_inverted = False

        # ---------------- Inversion detection added ----------------
        try:
            b, a = butter(4, (0.25, 25), 'bandpass', fs=fs)
            ecg_filt = filtfilt(b, a, epoch)
            ecg_clean = nk.ecg_clean(ecg_filt, sampling_rate=fs)
            flat_check = flatline_ratio(ecg_clean)
            if flat_check < 0.95 and np.std(ecg_clean) > 0:
                r = np.corrcoef(ecg_clean, np.abs(ecg_clean))[0, 1]
                inv_ratio = (1 - r) / 2
                try:
                    _, was_inverted = nk.ecg_invert(ecg_clean, sampling_rate=fs, show=False)
                except Exception:
                    was_inverted = np.nan
            else:
                inv_ratio = 0.0
                was_inverted = False
        except Exception:
            inv_ratio = np.nan
            was_inverted = np.nan
        # ------------------------------------------------------------

        try:
            hr_mean, hr_max, hr_min, hrv, snr_db = get_ecg_features(epoch, t_epoch, fs).tolist()
        except Exception:
            pass

        bad_clip = clip > th["clipping_max"]
        bad_flat = flat > th["flatline_max"]
        bad_miss = miss > th["missing_max"]
        bad_base = (not np.isnan(base)) and (base > th["baseline_max"])
        bad_hr = (np.isnan(hr_mean)) or (hr_mean < th["hr_min"]) or (hr_mean > th["hr_max"])
        bad_snr = (np.isnan(snr_db)) or (snr_db < th["snr_min"])
        bad_inv = (not np.isnan(inv_ratio)) and (inv_ratio > th["inv_ratio_max"])

        bad_epoch = bool(bad_clip or bad_flat or bad_miss or bad_base or bad_hr or bad_snr or bad_inv)

        results.append({
            "Epoch": int(i + 1),
            "Start_Time": str(abs_time[s]),
            "End_Time": str(abs_time[e - 1]),
            "Clipping_Ratio": float(clip),
            "Flatline_Ratio": float(flat),
            "Missing_Ratio": float(miss),
            "Baseline_Wander_Ratio": float(base) if not np.isnan(base) else None,
            "HR_Mean": float(hr_mean) if not np.isnan(hr_mean) else None,
            "SNR_dB": float(snr_db) if not np.isnan(snr_db) else None,
            "Inversion_Ratio": float(inv_ratio) if not np.isnan(inv_ratio) else None,   # new
            "Was_Inverted": bool(was_inverted) if not np.isnan(was_inverted) else None, # new
            "Bad_Epoch": bool(bad_epoch),
            "Bad_Clip": bool(bad_clip),
            "Bad_Flatline": bool(bad_flat),
            "Bad_Missing": bool(bad_miss),
            "Bad_Baseline": bool(bad_base),
            "Bad_HR": bool(bad_hr),
            "Bad_SNR": bool(bad_snr),
            "Bad_Inversion": bool(bad_inv),  # new
            "Raw_Data": epoch.tolist()
        })

    # --- per-metric summaries ---
    total = len(results)
    def ratio_summary(flag):
        bad_n = sum(r[flag] for r in results)
        good_n = total - bad_n
        return {
            "good_epochs": int(good_n),
            "bad_epochs": int(bad_n),
            "good_ratio": round(good_n / total, 3) if total else np.nan,
            "bad_ratio": round(bad_n / total, 3) if total else np.nan,
        }

    per_metric = {
        "Clipping": ratio_summary("Bad_Clip"),
        "Flatline": ratio_summary("Bad_Flatline"),
        "Missing": ratio_summary("Bad_Missing"),
        "Baseline": ratio_summary("Bad_Baseline"),
        "HR_Mean": ratio_summary("Bad_HR"),
        "SNR_dB": ratio_summary("Bad_SNR"),
        "Inversion": ratio_summary("Bad_Inversion"),  # new
    }

    overall_bad = sum(r["Bad_Epoch"] for r in results)
    overall_json = {
        "total_epochs": int(total),
        "good_epochs": int(total - overall_bad),
        "bad_epochs": int(overall_bad),
        "good_ratio": round((total - overall_bad) / total, 3) if total else np.nan,
        "bad_ratio": round(overall_bad / total, 3) if total else np.nan,
    }

    # --- Plotting ---
    def _shade(ax, flag):
        for r in results:
            start = np.datetime64(r["Start_Time"])
            end = np.datetime64(r["End_Time"])
            ax.axvspan(start, end, color=("red" if r[flag] else "green"), alpha=0.18)

    if plot in ("overall", "both"):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(abs_time, signal, lw=0.8, color="black")
        ax.set_title(f"{channel_name} — Overall QC (Red=Bad, Green=Good, incl. Inversion)")
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
            "Inversion_Ratio": "Bad_Inversion",  # new
        }
        for metric, flag in metric_flag_map.items():
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(abs_time, signal, lw=0.8, color="black")
            ax.set_title(f"{channel_name} — {metric} QC (Red=Bad, Green=Good)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True)
            _shade(ax, flag)
            plt.tight_layout()
            plt.show()

    return results, per_metric, overall_json