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
    Compute ECG features from raw ECG signal with physiological sanity checks.

    Returns
    -------
    array
        [HR_mean, HR_max, HR_min, RMSSD, SNR_dB]
        If HR or RMSSD is out of range → all values returned as NaN
    """

    # ---------------- Sanity thresholds (from your MATLAB logic) ----------------
    HR_MIN = 20      # bpm
    HR_MAX = 200     # bpm
    RMSSD_MIN = 2    # ms
    RMSSD_MAX = 300  # ms

    try:
        ecg = np.asarray(ecg, dtype=np.float64)

        # --- Bandpass filter ---
        b, a = butter(4, (0.25, 25), "bandpass", fs=fs)
        ecg_filt = filtfilt(b, a, ecg, axis=0)

        # --- Clean ECG ---
        ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)

        # --- R-peak detection ---
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="engzeemod2012")

    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    rr_times = time_in_sec[rpeaks["ECG_R_Peaks"]]
    if len(rr_times) < 3:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    # ---------------- HR ----------------
    d_rr = np.diff(rr_times)
    heart_rate = 60 / d_rr
    heart_rate = heart_rate[np.isfinite(heart_rate)]

    if heart_rate.size == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    hr_mean = np.mean(heart_rate)
    hr_min = np.min(heart_rate)
    hr_max = np.max(heart_rate)

    # ---------------- RMSSD ----------------
    d_rr_ms = d_rr * 1000
    d_d_rr_ms = np.diff(d_rr_ms)
    rmssd = np.sqrt(np.mean(d_d_rr_ms ** 2)) if len(d_d_rr_ms) > 0 else np.nan

    # ---------------- SANITY CHECKS ----------------
    invalid_epoch = False

    # HR sanity
    if (not np.isfinite(hr_mean)) or (hr_mean < HR_MIN) or (hr_mean > HR_MAX):
        invalid_epoch = True

    # RMSSD sanity
    if (not np.isfinite(rmssd)) or (rmssd < RMSSD_MIN) or (rmssd > RMSSD_MAX):
        invalid_epoch = True

    if invalid_epoch:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

    # ---------------- SNR ----------------
    ecg_with_rr = []
    ecg_with_rr_clean = []

    for rr in rr_times:
        idx = np.where((time_in_sec >= rr - 0.1) & (time_in_sec <= rr + 0.1))[0]
        if len(idx) > 0:
            ecg_with_rr.extend(ecg[idx])
            ecg_with_rr_clean.extend(ecg_cleaned[idx])

    ecg_with_rr = np.array(ecg_with_rr)
    ecg_with_rr_clean = np.array(ecg_with_rr_clean)

    signal_power = np.var(ecg_with_rr)
    noise_power = np.var(ecg_with_rr - ecg_with_rr_clean)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.nan

    return np.array([hr_mean, hr_max, hr_min, rmssd, snr_db])


# ---------------- QC runner: ANY fail => epoch BAD ----------------
def calculate_ecg_quality(channel_name, channel_dataframes, fs=200, epoch_len=30, thresholds=None, plot="overall"):

    th = {
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max": 0.50,
        "baseline_max": 0.30,
        "snr_min": 5.0,
        "inv_ratio_max": 0.5,
    }
    if thresholds:
        th.update(thresholds)

    samples_per_epoch = int(fs * epoch_len)
    df = channel_dataframes[channel_name]

    abs_time = np.array(df["Absolute Time"], dtype="datetime64[ns]")
    signal = np.array(df[channel_name], dtype=np.float64)

    mask_valid = ~np.isnan(signal)
    abs_time = abs_time[mask_valid]
    signal = signal[mask_valid]

    t0 = abs_time[0]
    time_in_sec = (abs_time - t0).astype("timedelta64[ns]").astype(float) / 1e9

    n_epochs = int(np.ceil(len(signal) / samples_per_epoch))

    # ---------- Metric Arrays ----------
    clipping_vals = np.zeros(n_epochs)
    flat_vals = np.zeros(n_epochs)
    missing_vals = np.zeros(n_epochs)
    baseline_vals = np.full(n_epochs, np.nan)
    hr_vals = np.full(n_epochs, np.nan)        # still computed, just not used for rejection
    snr_vals = np.full(n_epochs, np.nan)
    inv_vals = np.full(n_epochs, np.nan)

    # ---------- Metric Functions ----------
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

    def flatline_ratio(sig):
        return float(np.std(sig) < 1e-6)

    def missing_ratio(sig):
        return 1 - len(sig) / samples_per_epoch

    def baseline_ratio(signal, fs, cutoff=0.30):
        if len(signal) < fs:
            return np.nan
        f, pxx = welch(signal, fs=fs, nperseg=min(len(signal), 2048))
        total = float(np.sum(pxx))
        if total <= 0:
            return np.nan
        return float(np.sum(pxx[f <= cutoff]) / total)
    
    # ---------- Epoch Loop ----------
    epoch_start_times = []

    for i in range(n_epochs):
        s, e = i * samples_per_epoch, min((i + 1) * samples_per_epoch, len(signal))
        epoch = signal[s:e]
        t_epoch = time_in_sec[s:e]
        epoch_start_times.append(abs_time[s])

        clipping_vals[i] = clipping_ratio(epoch)
        flat_vals[i] = flatline_ratio(epoch)
        missing_vals[i] = missing_ratio(epoch)
        baseline_vals[i] = baseline_ratio(epoch, fs, cutoff=0.30)

        try:
            hr_mean, hr_max, hr_min, hrv, snr_db = get_ecg_features(epoch, t_epoch, fs)
            hr_vals[i] = hr_mean
            snr_vals[i] = snr_db
        except:
            pass

        # --- Inversion detection (skip if flatline) ---
        if flat_vals[i] <= th["flatline_max"]:  # only attempt if NOT flat
            try:
                ecg_clean = nk.ecg_clean(epoch, sampling_rate=fs)
                _, inverted = nk.ecg_invert(ecg_clean, sampling_rate=fs)
                inv_vals[i] = 1.0 if inverted else 0.0
            except:
                inv_vals[i] = np.nan
        else:
            inv_vals[i] = np.nan


    epoch_start_times = np.array(epoch_start_times)

    # ---------- Boolean Masks ----------
    masks = {
        "clipping": clipping_vals > th["clipping_max"],
        "flatline": flat_vals > th["flatline_max"],
        "missing": missing_vals > th["missing_max"],
        "baseline": baseline_vals > th["baseline_max"],
        "snr": (snr_vals < th["snr_min"]) | np.isnan(snr_vals),
        "inversion": inv_vals > th["inv_ratio_max"],
    }

    combined_mask = np.any(np.column_stack(list(masks.values())), axis=1)

    # ---------- Plotting (Red = bad, Green = good) ----------
    def shade_epochs(ax, bad_mask):
        for i, bad in enumerate(bad_mask):
            start = epoch_start_times[i]
            end = start + np.timedelta64(int(epoch_len), "s")
            color = "red" if bad else "green"
            ax.axvspan(start, end, color=color, alpha=0.18)

    if plot in ("overall", "both"):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(abs_time, signal, lw=0.8, color="black")
        ax.set_title(f"{channel_name} — Overall QC (Green = Good, Red = Bad)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True)
        shade_epochs(ax, combined_mask)
        plt.tight_layout()
        plt.show()

    if plot in ("per-metric", "both"):
        for name, mask in masks.items():
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(abs_time, signal, lw=0.8, color="black")
            ax.set_title(f"{channel_name} — {name.upper()} QC")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True)
            shade_epochs(ax, mask)
            plt.tight_layout()
            plt.show()

    return {
        "metric_names": ["clipping", "flatline", "missing", "baseline", "hr_mean", "snr_db", "inversion"],
        "metric_values": {
            "clipping": clipping_vals,
            "flatline": flat_vals,
            "missing": missing_vals,
            "baseline": baseline_vals,
            "hr_mean": hr_vals,   # returned but not used for QC decision
            "snr_db": snr_vals,
            "inversion": inv_vals,
        },
        "bad_masks": masks,
        "combined_mask": combined_mask,
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channel": channel_name
        }
    }