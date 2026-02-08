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
import time


def calculate_emg_quality(signal, sampling_rate, channel_names=None,
                          epoch_len=30, plot=False):
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
        If True, show spectrogram-style visualization with bad epochs marked.

    Returns
    -------
    dict
        Dictionary containing:
        - Per-epoch quality masks
        - EMG power metrics
        - Visualization array
        - Metadata
    """

    # -------------------------------
    # Basic dimensions and reshaping
    # -------------------------------

    fs = sampling_rate                                  # Sampling frequency (Hz)
    epoch_samps = int(fs * epoch_len)                   # Samples per epoch
    n_ch, n_samples = signal.shape                      # Channel and sample counts
    n_epochs = n_samples // epoch_samps                 # Number of full epochs

    # Truncate signal to an integer number of epochs
    # and reshape to (channel × epoch × samples)
    X = signal[:, :n_epochs * epoch_samps].astype(np.float32, copy=False)
    X = X.reshape(n_ch, n_epochs, epoch_samps)

    # -------------------------------
    # Frequency-domain setup
    # -------------------------------

    # FFT frequency vector (0 → Nyquist)
    freqs = rfftfreq(epoch_samps, 1 / fs)

    # EMG physiological band:
    # EMG is typically considered meaningful above ~10 Hz
    emg_band = (freqs >= 10) & (freqs <= 100)

    # Frequencies to visualize (high enough to see EMG noise/clipping)
    plot_mask = freqs <= 125

    # -------------------------------
    # FFT computation
    # -------------------------------

    # Compute magnitude spectrum for each epoch
    # Shape: (channel × epoch × frequency)
    F = np.abs(rfft(X, axis=2))

    # Log power used only for visualization (not QC decisions)
    logpow = np.log10(F + 1e-12)

    # -------------------------------
    # Spectral EMG power metric
    # -------------------------------

    # Mean EMG-band power per epoch
    # Used ONLY to detect dead / disconnected channels
    emg_power = F[..., emg_band].mean(axis=2)

    # -------------------------------
    # Time-domain integrity metrics
    # -------------------------------

    # Variance per epoch (flatline detection)
    epoch_var = np.var(X, axis=2)

    # Peak-to-peak amplitude per epoch
    epoch_ptp = np.ptp(X, axis=2)

    # Difference between adjacent samples
    # Used to detect repeated values (digital flatline)
    diffs = np.diff(X, axis=2)

    # Fraction of samples that do not change
    repeat_ratio = np.mean(np.abs(diffs) < 1e-6, axis=2)

    # =========================================================
    # ABSOLUTE THRESHOLDS (NO PERCENTILES, NO NORMALIZATION)
    # =========================================================

    # Flatline thresholds
    FLAT_VAR_THRESH = 1e-12      # Essentially zero variance
    FLAT_PTP_THRESH = 1e-6       # Essentially zero amplitude

    # Low-power threshold
    # Interpreted as "dead EMG channel" rather than artifact
    LOW_POWER_THRESH = 500       # Units depend on EDF scaling (µV typical)

    # -------------------------------
    # Flatline / disconnection mask
    # -------------------------------

    flat_mask = (
        (epoch_var < FLAT_VAR_THRESH) |      # No variance
        (epoch_ptp < FLAT_PTP_THRESH) |      # No amplitude
        (repeat_ratio > 0.98)                # Digital repetition
    )

    # ---------------------------------------------------------
    # Saturation / clipping detection (ADC railing)
    # ---------------------------------------------------------
    # This detects epochs where the signal is pinned near its
    # min or max value for a sustained fraction of time.
    #
    # This is NOT "large EMG" — it is loss of dynamic range.

    CLIP_FRAC_THRESH = 0.05     # ≥5% of samples stuck at rail
    CLIP_EPS_FRAC = 1e-4        # "Near rail" = within 0.01% of epoch range

    # Per-epoch min, max, and range
    epoch_min = X.min(axis=2, keepdims=True)
    epoch_max = X.max(axis=2, keepdims=True)
    epoch_rng = epoch_max - epoch_min

    # Numerical tolerance for detecting plateaus
    eps = np.maximum(epoch_rng * CLIP_EPS_FRAC, 1e-12)

    # Fraction of samples near min and max rails
    near_min = np.mean(X <= (epoch_min + eps), axis=2)
    near_max = np.mean(X >= (epoch_max - eps), axis=2)

    # Saturation occurs if a sustained plateau exists at either rail
    saturation_mask = (
        (near_min > CLIP_FRAC_THRESH) |
        (near_max > CLIP_FRAC_THRESH)
    )

    # -------------------------------
    # Low-power (dead EMG) mask
    # -------------------------------

    low_power_mask = emg_power < LOW_POWER_THRESH

    # -------------------------------
    # Final combined bad-epoch mask
    # -------------------------------

    bad_mask = flat_mask | saturation_mask | low_power_mask

    # =========================================================
    # Visualization output (always computed)
    # =========================================================

    # Use only valid epochs to compute z-score normalization
    valid_mask = ~bad_mask
    valid_vals = logpow[valid_mask]

    mean = np.nanmean(valid_vals, axis=0, keepdims=True)
    std = np.nanstd(valid_vals, axis=0, keepdims=True)

    # Z-score log power for visualization only
    logpow_z = (logpow - mean) / (std + 1e-8)

    # Force bad epochs to a very low value for visual clarity
    logpow_z[bad_mask] = -5

    # Light smoothing across time and frequency
    smooth = gaussian_filter(logpow_z, sigma=(0, 1, 1))[:, :, plot_mask]

    # Final visualization array:
    # (channel × frequency × epoch)
    visual_output = np.transpose(smooth, (0, 2, 1))

    # =========================================================
    # Optional plotting
    # =========================================================

    if plot:

        fig, axes = plt.subplots(
            n_ch, 1,
            figsize=(20, 2 * n_ch),
            sharex=True
        )

        if n_ch == 1:
            axes = [axes]

        # X-axis ticks every 30 minutes
        tick_interval_epochs = max(1, int((30 * 60) // epoch_len))
        tick_positions = np.arange(0, n_epochs, tick_interval_epochs)
        tick_labels = [
            time.strftime('%H:%M', time.gmtime(t * epoch_len))
            for t in tick_positions
        ]

        # Frequency axis labels (true Hz)
        visual_freqs_hz = freqs[plot_mask]
        max_hz = visual_freqs_hz[-1]

        desired_ticks_hz = np.array([0, 25, 50, 75, 100, 125])
        desired_ticks_hz = desired_ticks_hz[desired_ticks_hz <= max_hz]

        freq_tick_inds = [
            np.argmin(np.abs(visual_freqs_hz - hz))
            for hz in desired_ticks_hz
        ]

        for ci, ch in enumerate(channel_names or [f"ch{ci}" for ci in range(n_ch)]):
            ax = axes[ci]
            S = visual_output[ci]

            ax.imshow(
                S,
                aspect="auto",
                origin="lower",
                cmap="jet",
                vmin=-2,
                vmax=2
            )

            # Overlay bad epochs
            for e in np.where(bad_mask[ci])[0]:
                ax.axvspan(e, e + 1, color="magenta", alpha=0.5, lw=0)

            ax.set_ylabel(ch)
            ax.set_yticks(freq_tick_inds)
            ax.set_yticklabels([f"{hz:g}" for hz in desired_ticks_hz])

            if ci == n_ch - 1:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
                ax.set_xlabel("Time (HH:MM)")

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
            "visual_freqs_hz": freqs[plot_mask],
            "visual_epoch_times_s": np.arange(n_epochs) * epoch_len
        }
    }





