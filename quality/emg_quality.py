# -*- coding: utf-8 -*-
"""
EMG Quality Control for Sleep EDFs

-------
This script performs absolute, physics-based EMG quality control suitable
for sleep staging. It flags EMG epochs that are unusable due to:

- Flatline / disconnection
- ADC saturation / clipping
- Extremely low signal power (dead channel)

Importantly:
- NO percentile-based thresholds are used
- NO spectral-shape heuristics are used
- All criteria are absolute and interpretable

The output includes:
- Per-epoch bad masks
- Visualization-ready spectrogram output
- Metadata for downstream QC or reporting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch
import time


def _butter_bandpass(low_hz, high_hz, fs, order=4):
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    if low >= high:
        raise ValueError(f"Invalid bandpass: low={low_hz}Hz high={high_hz}Hz for fs={fs}")
    b, a = butter(order, [low, high], btype="band")
    return b, a


def _apply_notch(x, fs, notch_hz=60.0, q=30.0):
    # iirnotch expects normalized frequency (w0 = f0/(fs/2))
    w0 = notch_hz / (fs / 2.0)
    if w0 <= 0 or w0 >= 1:
        return x
    b, a = iirnotch(w0=w0, Q=q)
    return filtfilt(b, a, x).astype(np.float32, copy=False)


def _emg_preprocess_for_view(x, fs,
                            bandpass=(10.0, 100.0),
                            notch_hz=60.0,
                            notch_q=30.0,
                            add_120hz_notch=True,
                            demean=True):
    """
    Typical PSG-style EMG visualization preprocessing:
    - (Optional) demean
    - bandpass 10–100 Hz
    - notch at 60 Hz (+ optionally 120 Hz if within Nyquist)
    """
    y = x.astype(np.float32, copy=False)

    if demean:
        y = y - np.nanmean(y)

    # Bandpass
    b, a = _butter_bandpass(bandpass[0], bandpass[1], fs, order=4)
    y = filtfilt(b, a, np.nan_to_num(y)).astype(np.float32, copy=False)

    # Notch 60 Hz
    if notch_hz is not None:
        y = _apply_notch(y, fs, notch_hz=notch_hz, q=notch_q)

    # Optional 120 Hz notch (only if representable)
    if add_120hz_notch and (fs / 2.0) > 125:
        y = _apply_notch(y, fs, notch_hz=120.0, q=notch_q)

    return y


def _running_rms(x, fs, win_s=0.2):
    """
    Short-window RMS envelope (good for visualizing EMG tone/bursts).
    """
    n = int(max(1, round(win_s * fs)))
    # Use convolution on squared signal
    x2 = (x.astype(np.float32) ** 2)
    kernel = np.ones(n, dtype=np.float32) / n
    ma = np.convolve(x2, kernel, mode="same")
    return np.sqrt(np.maximum(ma, 0.0)).astype(np.float32, copy=False)


def calculate_emg_quality(signal, sampling_rate, channel_names=None,
                          epoch_len=30, plot=False,
                          # --- plotting controls ---
                          plot_seconds_per_screen=None,   # None = full-night view
                          plot_decimate_to_hz=200.0,       # decimate trace for speed (visual only)
                          view_bandpass=(10.0, 100.0),
                          view_notch_hz=60.0,
                          view_envelope_rms_s=0.2):
    """
    Perform EMG quality assessment on sleep data.

    Parameters
    ----------
    signal : np.ndarray
        EMG signal array of shape (n_channels, n_samples).
        Expected to be in physical units (usually µV from EDF).
    sampling_rate : int
        Sampling rate in Hz.
    channel_names : list of str or None
        Channel labels (for plotting and metadata).
    epoch_len : float
        Epoch length in seconds (default = 30, standard sleep epoch).
    plot : bool
        If True, show time-domain EMG trace per channel with bad epochs marked.

    Returns
    -------
    dict
        Dictionary containing:
        - Per-epoch quality masks
        - EMG power metrics
        - Visualization array (kept for compatibility, but not used for plotting)
        - Metadata
    """

    # -------------------------------
    # Basic dimensions and reshaping
    # -------------------------------
    fs = sampling_rate
    epoch_samps = int(fs * epoch_len)
    n_ch, n_samples = signal.shape
    n_epochs = n_samples // epoch_samps

    # Truncate signal to an integer number of epochs
    X = signal[:, :n_epochs * epoch_samps].astype(np.float32, copy=False)
    X = X.reshape(n_ch, n_epochs, epoch_samps)

    # -------------------------------
    # Frequency-domain setup (kept for your QC metric)
    # -------------------------------
    freqs = rfftfreq(epoch_samps, 1 / fs)

    # EMG physiological band (for dead-channel detection)
    emg_band = (freqs >= 10) & (freqs <= 100)

    # -------------------------------
    # FFT computation (for emg_power metric only)
    # -------------------------------
    F = np.abs(rfft(X, axis=2))
    emg_power = F[..., emg_band].mean(axis=2)

    # -------------------------------
    # Time-domain integrity metrics
    # -------------------------------
    epoch_var = np.var(X, axis=2)
    epoch_ptp = np.ptp(X, axis=2)
    diffs = np.diff(X, axis=2)
    repeat_ratio = np.mean(np.abs(diffs) < 1e-6, axis=2)

    # =========================================================
    # ABSOLUTE THRESHOLDS (NO PERCENTILES, NO NORMALIZATION)
    # =========================================================
    FLAT_VAR_THRESH = 1e-12
    FLAT_PTP_THRESH = 1e-6

    # NOTE: this threshold is in "mean FFT magnitude in 10–100 Hz"
    # and is EDF scaling dependent. You already had it at 500.
    LOW_POWER_THRESH = 500

    flat_mask = (
        (epoch_var < FLAT_VAR_THRESH) |
        (epoch_ptp < FLAT_PTP_THRESH) |
        (repeat_ratio > 0.98)
    )

    # ---------------------------------------------------------
    # Saturation / clipping detection (ADC railing)
    # ---------------------------------------------------------
    CLIP_FRAC_THRESH = 0.05
    CLIP_EPS_FRAC = 1e-4

    epoch_min = X.min(axis=2, keepdims=True)
    epoch_max = X.max(axis=2, keepdims=True)
    epoch_rng = epoch_max - epoch_min
    eps = np.maximum(epoch_rng * CLIP_EPS_FRAC, 1e-12)

    near_min = np.mean(X <= (epoch_min + eps), axis=2)
    near_max = np.mean(X >= (epoch_max - eps), axis=2)

    saturation_mask = (
        (near_min > CLIP_FRAC_THRESH) |
        (near_max > CLIP_FRAC_THRESH)
    )

    low_power_mask = emg_power < LOW_POWER_THRESH
    bad_mask = flat_mask | saturation_mask | low_power_mask

    # =========================================================
    # Sample-level visualization output 
    # =========================================================
    n_samples_trunc = n_epochs * epoch_samps
    X_cont = signal[:, :n_samples_trunc].astype(np.float32, copy=False)
    
    visual_output = np.zeros((n_ch, 2, n_samples_trunc), dtype=np.float32)
    
    for ci in range(n_ch):
        # Blue
        y = _emg_preprocess_for_view(
            X_cont[ci],
            fs=fs,
            bandpass=view_bandpass,
            notch_hz=view_notch_hz,
            notch_q=30.0,
            add_120hz_notch=True,
            demean=True
        )
    
        # Orange
        env = _running_rms(np.abs(y), fs=fs, win_s=view_envelope_rms_s)
    
        visual_output[ci, 0, :] = y
        visual_output[ci, 1, :] = env
    
    # Hard check so it can't silently flatten
    assert visual_output.ndim == 3, visual_output.shape
    assert visual_output.shape == (n_ch, 2, n_samples_trunc), visual_output.shape


    # =========================================================
    # Optional plotting: TIME-DOMAIN EMG TRACE (PSG-style)
    # =========================================================
    if plot:
        # Flatten back to continuous per channel
        X_cont = signal[:, :n_epochs * epoch_samps].astype(np.float32, copy=False)

        # Decimate for plotting speed (visual only)
        if plot_decimate_to_hz is not None and plot_decimate_to_hz > 0 and fs > plot_decimate_to_hz:
            decim = int(np.floor(fs / plot_decimate_to_hz))
        else:
            decim = 1

        fs_plot = fs / decim
        t = np.arange(X_cont.shape[1], dtype=np.float32) / fs
        t_plot = t[::decim]

        # Optional windowing for "screen" view
        if plot_seconds_per_screen is not None:
            max_s = float(plot_seconds_per_screen)
            keep = t_plot <= max_s
        else:
            keep = slice(None)

        fig, axes = plt.subplots(
            n_ch, 1,
            figsize=(18, 2.0 * n_ch),
            sharex=True
        )
        if n_ch == 1:
            axes = [axes]

        # X ticks every 30 minutes (same style as before)
        tick_interval_epochs = max(1, int((30 * 60) // epoch_len))
        tick_positions_epochs = np.arange(0, n_epochs, tick_interval_epochs)
        tick_positions_s = tick_positions_epochs * epoch_len
        tick_labels = [
            time.strftime('%H:%M', time.gmtime(tsec))
            for tsec in tick_positions_s
        ]

        # Precompute shaded spans in seconds for each channel
        # (we shade per-channel because bad_mask is per-channel)
        for ci, ch in enumerate(channel_names or [f"ch{ci}" for ci in range(n_ch)]):
            ax = axes[ci]

            # PSG-style viewing preprocess
            y = _emg_preprocess_for_view(
                X_cont[ci],
                fs=fs,
                bandpass=view_bandpass,
                notch_hz=view_notch_hz,
                notch_q=30.0,
                add_120hz_notch=True,
                demean=True
            )

            # Envelope (RMS)
            env = _running_rms(np.abs(y), fs=fs, win_s=view_envelope_rms_s)

            # Decimate for plotting
            y_plot = y[::decim][keep]
            env_plot = env[::decim][keep]
            tp = t_plot[keep]

            ax.plot(tp, y_plot, lw=0.6)
            ax.plot(tp, env_plot, lw=1.0)

            # Shade bad epochs
            bad_epochs = np.where(bad_mask[ci])[0]
            for e in bad_epochs:
                start = e * epoch_len
                end = (e + 1) * epoch_len
                # Only shade if within view
                if plot_seconds_per_screen is not None and start > plot_seconds_per_screen:
                    break
                ax.axvspan(start, end, color="magenta", alpha=0.25, lw=0)

            ax.set_ylabel(ch)
            ax.grid(True, alpha=0.15)

        # X-axis formatting
        axes[-1].set_xlabel("Time (s)")
        if plot_seconds_per_screen is None:
            # full-night: label as HH:MM at 30-min intervals (in seconds)
            axes[-1].set_xticks(tick_positions_s)
            axes[-1].set_xticklabels(tick_labels)
            axes[-1].set_xlabel("Time (HH:MM)")

        plt.tight_layout()
        plt.show()

    # =========================================================
    # Return structured output
    # =========================================================
    return {
        "metric_names": ["emg_power"],
        "metric_values": {
            "emg_power": emg_power,
        },
        "flat_mask": flat_mask,
        "saturation_mask": saturation_mask,
        "low_power_mask": low_power_mask,
        "combined_flags": bad_mask,
        "visual_output": visual_output,
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channels": channel_names,
            "visual_epoch_times_s": np.arange(n_epochs) * epoch_len
        }
    }
