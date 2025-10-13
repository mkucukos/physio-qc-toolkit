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