import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt
import neurokit2 as nk

# ============================================================
# FAST HELPERS
# ============================================================

def baseline_ratio_fast(sig, fs):
    if len(sig) < fs:
        return np.nan
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    total = psd.sum()
    return psd[freqs <= 0.30].sum() / total if total > 0 else np.nan


def clipping_ratio(sig):
    if sig.size < 10:
        return np.nan
    s_min, s_max = np.nanmin(sig), np.nanmax(sig)
    rng = s_max - s_min
    if rng <= 1e-12:
        return 1.0
    eps = rng * 0.001
    return max(
        np.mean(sig <= s_min + eps),
        np.mean(sig >= s_max - eps),
    )

# ============================================================
# OPTIMIZED ECG QUALITY FUNCTION
# ============================================================

def calculate_ecg_quality(
    channel_name,
    channel_dataframes,
    fs=200,
    epoch_len=30,
    thresholds=None,
    plot=False,
):

    # ---------------- Thresholds ----------------
    th = {
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max": 0.50,
        "baseline_max": 0.30,
        "snr_min": 5.0,
        "inv_ratio_max": 0.5,
        "HR_MIN": 20,
        "HR_MAX": 200,
        "RMSSD_MIN": 2,
        "RMSSD_MAX": 300,
    }
    if thresholds:
        th.update(thresholds)

    samples_per_epoch = int(fs * epoch_len)
    df = channel_dataframes[channel_name]

    abs_time = np.array(df["Absolute Time"], dtype="datetime64[ns]")
    signal = np.array(df[channel_name], dtype=np.float64)

    valid = ~np.isnan(signal)
    abs_time = abs_time[valid]
    signal = signal[valid]

    if signal.size == 0:
        return None

    t0 = abs_time[0]
    time_in_sec = (abs_time - t0).astype("timedelta64[ns]").astype(float) / 1e9
    n_epochs = int(np.ceil(len(signal) / samples_per_epoch))

    # ---------------- GLOBAL PREPROCESSING ----------------
    b, a = butter(4, (0.25, 25), "bandpass", fs=fs)
    signal_filt = filtfilt(b, a, signal)
    signal_clean = nk.ecg_clean(signal_filt, sampling_rate=fs)

    try:
        _, rpeaks = nk.ecg_peaks(signal_clean, sampling_rate=fs)
        rpeak_times = time_in_sec[rpeaks["ECG_R_Peaks"]]
    except Exception:
        rpeak_times = np.array([])

    # ---------------- STORAGE ----------------
    clipping_vals = np.zeros(n_epochs)
    flat_vals = np.zeros(n_epochs)
    missing_vals = np.zeros(n_epochs)

    baseline_vals = np.full(n_epochs, np.nan)
    hr_vals = np.full(n_epochs, np.nan)
    snr_vals = np.full(n_epochs, np.nan)
    inv_vals = np.full(n_epochs, np.nan)

    baseline_skipped = np.zeros(n_epochs, dtype=bool)
    snr_skipped = np.zeros(n_epochs, dtype=bool)
    inv_skipped = np.zeros(n_epochs, dtype=bool)

    epoch_start_times = []

    # ---------------- EPOCH LOOP ----------------
    for i in range(n_epochs):
        s = i * samples_per_epoch
        e = min((i + 1) * samples_per_epoch, len(signal))

        epoch = signal[s:e]
        epoch_clean = signal_clean[s:e]
        t_epoch = time_in_sec[s:e]

        epoch_start_times.append(abs_time[s])

        clipping_vals[i] = clipping_ratio(epoch)
        flat_vals[i] = float(np.std(epoch) < 1e-6)
        missing_vals[i] = 1 - len(epoch) / samples_per_epoch

        usable = (
            flat_vals[i] <= th["flatline_max"] and
            clipping_vals[i] <= th["clipping_max"] and
            missing_vals[i] <= th["missing_max"]
        )

        # ---------- Baseline / HR / SNR ----------
        if usable:
            baseline_vals[i] = baseline_ratio_fast(epoch, fs)

            epoch_rr = rpeak_times[
                (rpeak_times >= t_epoch[0]) &
                (rpeak_times <= t_epoch[-1])
            ]

            if len(epoch_rr) >= 3:
                d_rr = np.diff(epoch_rr)
                hr = 60 / d_rr
                hr = hr[np.isfinite(hr)]

                if hr.size:
                    hr_mean = hr.mean()
                    d_rr_ms = d_rr * 1000
                    rmssd = np.sqrt(np.mean(np.diff(d_rr_ms) ** 2))

                    if (
                        th["HR_MIN"] <= hr_mean <= th["HR_MAX"] and
                        th["RMSSD_MIN"] <= rmssd <= th["RMSSD_MAX"]
                    ):
                        hr_vals[i] = hr_mean

                rr_mask = np.abs(t_epoch[:, None] - epoch_rr[None, :]) <= 0.1
                if rr_mask.any():
                    ecg_rr = epoch[rr_mask.any(axis=1)]
                    ecg_rr_clean = epoch_clean[rr_mask.any(axis=1)]
                    noise = ecg_rr - ecg_rr_clean
                    if np.var(noise) > 0:
                        snr_vals[i] = 10 * np.log10(
                            np.var(ecg_rr) / np.var(noise)
                        )
        else:
            baseline_skipped[i] = True
            snr_skipped[i] = True

        # ---------- Inversion (STRICT GATE) ----------
        if usable:
            try:
                _, inverted = nk.ecg_invert(epoch_clean, sampling_rate=fs)
                inv_vals[i] = 1.0 if inverted else 0.0
            except Exception:
                inv_vals[i] = np.nan
        else:
            inv_skipped[i] = True
            inv_vals[i] = np.nan

    epoch_start_times = np.array(epoch_start_times)

    # ---------------- MASKS ----------------
    masks = {
        "clipping": clipping_vals > th["clipping_max"],
        "flatline": flat_vals > th["flatline_max"],
        "missing": missing_vals > th["missing_max"],
        "baseline": (~baseline_skipped) & (baseline_vals > th["baseline_max"]),
        "snr": (~snr_skipped) & ((snr_vals < th["snr_min"]) | np.isnan(snr_vals)),
        "inversion": (~inv_skipped) & (inv_vals > th["inv_ratio_max"]),
    }

    combined_mask = np.any(np.column_stack(list(masks.values())), axis=1)

    # ---------------- PLOTTING ----------------
    if plot:
        def shade(ax, bad_mask, skip_mask=None):
            for i in range(len(epoch_start_times)):
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
            figsize=(16, 2 * (len(masks) + 1)),
            sharex=True,
        )

        axes[0].plot(abs_time, signal, lw=0.8, color="black")
        shade(axes[0], combined_mask)
        axes[0].set_title(f"{channel_name} — Overall ECG QC")

        for ax, name in zip(axes[1:], masks):
            ax.plot(abs_time, signal, lw=0.8, color="black")

            if name in ("baseline", "snr"):
                shade(ax, masks[name], baseline_skipped | snr_skipped)
            elif name == "inversion":
                shade(ax, masks[name], inv_skipped)
            else:
                shade(ax, masks[name])

            ax.set_title(name.upper())

        for ax in axes:
            ax.set_xlim(abs_time[0], abs_time[-1])

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    # ---------------- RETURN ----------------
    return {
        "metric_names": list(masks.keys()),
        "metric_values": {
            "clipping": clipping_vals,
            "flatline": flat_vals,
            "missing": missing_vals,
            "baseline": baseline_vals,
            "hr_mean": hr_vals,
            "snr_db": snr_vals,
            "inversion": inv_vals,
        },
        "bad_masks": masks,
        "combined_mask": combined_mask,
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channel": channel_name,
        },
    }