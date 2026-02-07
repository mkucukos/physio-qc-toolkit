# -*- coding: utf-8 -*-
"""
EOG quality assessment for sleep EDFs.

- Exactly 2 EOG channels are REQUIRED as input (we always compute cross-channel correlation).
- Bad epoch flags are based on per-channel, absolute, staging-relevant failure modes:
  1) flatline / disconnection
  2) clipping / saturation (plateau at rail)
  3) large "pops" (huge slope events; loose electrode / cable bump)
  4) optional HF contamination (kept here as an absolute check; you can disable easily)

- Cross-channel correlation is computed and plotted as DIAGNOSTIC ONLY.
  We do NOT reject epochs based on low or high correlation, because for staging you can
  often keep one good channel when the other is bad.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter
import time


def calculate_eog_quality(signal, sampling_rate, channel_names=None,
                          epoch_len=30, plot=False):
    """
    Parameters
    ----------
    signal : np.ndarray
        EOG array of shape (2, n_samples). MUST be exactly 2 channels.
    sampling_rate : int | float
        Sampling rate in Hz.
    channel_names : list[str] | None
        Length-2 labels for plotting/metadata.
    epoch_len : float
        Epoch length in seconds (default 30).
    plot : bool
        If True, plot spectrogram-like panels for each EOG + a correlation trace.

    Returns
    -------
    dict
        Contains masks, metrics, visual_output (2 x freq x epoch), and metadata.
    """

    # -------------------------------
    # Input requirements
    # -------------------------------
    fs = float(sampling_rate)

    if signal.ndim != 2:
        raise ValueError("signal must be 2D: (n_ch, n_samples)")

    n_ch, n_samples = signal.shape
    if n_ch != 2:
        raise ValueError(f"EOG QC requires exactly 2 channels; got n_ch={n_ch}")

    if channel_names is None:
        channel_names = ["EOG1", "EOG2"]
    if len(channel_names) != 2:
        raise ValueError("channel_names must be length 2 for EOG QC")

    # -------------------------------
    # Epoching (2, epoch, samples)
    # -------------------------------
    epoch_samps = int(fs * epoch_len)
    n_epochs = n_samples // epoch_samps

    # Truncate to full epochs only; reshape for vectorized per-epoch operations
    X = signal[:, :n_epochs * epoch_samps].astype(np.float32, copy=False)
    X = X.reshape(2, n_epochs, epoch_samps)

    # -------------------------------
    # FFT + visualization frequency range
    # EOG is mostly < ~15 Hz physiologically, but plotting to 30 Hz helps see contamination.
    # -------------------------------
    freqs = rfftfreq(epoch_samps, 1 / fs)
    plot_mask = freqs <= 30.0

    # EOG band for optional spectral checks (not required for staging; kept as a sanity metric)
    eog_band = (freqs >= 0.3) & (freqs <= 15.0)
    hf_band = (freqs > 15.0) & (freqs <= min(30.0, fs / 2.0))

    # -------------------------------
    # FFT / log-power (used for visualization output only)
    # -------------------------------
    F = np.abs(rfft(X, axis=2))  # (2, epoch, freq)
    logpow = np.log10(F + 1e-12)

    # -------------------------------
    # Time-domain integrity metrics (per channel, per epoch)
    # -------------------------------
    epoch_var = np.var(X, axis=2)
    epoch_ptp = np.ptp(X, axis=2)

    diffs = np.diff(X, axis=2)
    repeat_ratio = np.mean(np.abs(diffs) < 1e-6, axis=2)

    # -------------------------------
    # Cross-channel correlation (DIAGNOSTIC ONLY)
    # We remove DC within each epoch because EOG channels commonly drift.
    # This yields a per-epoch correlation time series of shape (n_epochs,).
    # -------------------------------
    X0 = X - X.mean(axis=2, keepdims=True)
    num = np.sum(X0[0] * X0[1], axis=1)
    den = np.sqrt(np.sum(X0[0] ** 2, axis=1) * np.sum(X0[1] ** 2, axis=1)) + 1e-12
    eog_corr = num / den

    # -------------------------------
    # Absolute thresholds (NO percentiles)
    # -------------------------------
    FLAT_VAR_THRESH = 1e-12
    FLAT_PTP_THRESH = 1e-6
    REPEAT_RATIO_THRESH = 0.98

    # Clipping detection (plateau at epoch min/max)
    CLIP_FRAC_THRESH = 0.05  # >=5% of samples pinned to a rail
    CLIP_EPS_FRAC = 1e-4     # "near rail" = within 0.01% of epoch range

    # Pops (electrode movement / cable bump): huge slope, not physiologic EOG
    # Units are "signal units per second". If EDF is in µV, this is µV/s.
    POP_SLOPE_UV_PER_S = 50000.0

    # Optional HF contamination ratio (kept as an absolute flag; set HF_RATIO_THRESH = np.inf to disable)
    HF_RATIO_THRESH = 1.5

    # -------------------------------
    # Masks: flatline / dead lead
    # -------------------------------
    flat_mask = (
        (epoch_var < FLAT_VAR_THRESH) |
        (epoch_ptp < FLAT_PTP_THRESH) |
        (repeat_ratio > REPEAT_RATIO_THRESH)
    )

    # -------------------------------
    # Masks: saturation/clipping via plateau detection
    # This detects "railing" where the signal pins near min or max for a sustained fraction.
    # -------------------------------
    epoch_min = X.min(axis=2, keepdims=True)
    epoch_max = X.max(axis=2, keepdims=True)
    epoch_rng = epoch_max - epoch_min
    eps = np.maximum(epoch_rng * CLIP_EPS_FRAC, 1e-12)

    near_min = np.mean(X <= (epoch_min + eps), axis=2)
    near_max = np.mean(X >= (epoch_max - eps), axis=2)
    saturation_mask = (near_min > CLIP_FRAC_THRESH) | (near_max > CLIP_FRAC_THRESH)

    # -------------------------------
    # Masks: electrode pops (large slope)
    # -------------------------------
    max_abs_diff = np.max(np.abs(diffs), axis=2)  # units per sample
    max_slope = max_abs_diff * fs                 # units per second
    pop_mask = max_slope > POP_SLOPE_UV_PER_S

    # -------------------------------
    # Optional: HF contamination ratio (absolute)
    # You can disable by setting HF_RATIO_THRESH very large or by forcing hf_mask = False.
    # -------------------------------
    eog_power = F[..., eog_band].mean(axis=2)
    hf_power = F[..., hf_band].mean(axis=2) if np.any(hf_band) else np.zeros_like(eog_power)
    hf_ratio = hf_power / (eog_power + 1e-12)
    hf_mask = hf_ratio > HF_RATIO_THRESH

    # -------------------------------
    # Final combined bad mask (per channel, per epoch)
    # NOTE: correlation is NOT used for rejection.
    # -------------------------------
    bad_mask = flat_mask | saturation_mask | pop_mask | hf_mask

    # -------------------------------
    # Visualization output (always computed): z-scored log power, smoothed
    # -------------------------------
    valid_mask = ~bad_mask
    valid_vals = logpow[valid_mask]

    mean = np.nanmean(valid_vals, axis=0, keepdims=True)
    std = np.nanstd(valid_vals, axis=0, keepdims=True)

    logpow_z = (logpow - mean) / (std + 1e-8)

    # Push flagged epochs down so they are visually obvious
    logpow_z[bad_mask] = -5

    smooth = gaussian_filter(logpow_z, sigma=(0, 1, 1))[:, :, plot_mask]
    visual_output = np.transpose(smooth, (0, 2, 1))  # (2, freq, epoch)

    # -------------------------------
    # Optional plotting:
    # - Two spectrogram rows (one per EOG channel)
    # - One correlation row (epoch-level diagnostic)
    # -------------------------------
    if plot:

        fig, axes = plt.subplots(3, 1, figsize=(18, 6), sharex=True)

        # X-axis ticks every 30 minutes
        tick_interval_epochs = max(1, int((30 * 60) // epoch_len))
        tick_positions = np.arange(0, n_epochs, tick_interval_epochs)
        tick_labels = [time.strftime('%H:%M', time.gmtime(t * epoch_len)) for t in tick_positions]

        # Y-axis frequency ticks (true Hz values from FFT)
        visual_freqs_hz = freqs[plot_mask]
        max_hz = visual_freqs_hz[-1]
        desired_ticks_hz = np.array([0, 5, 10, 15, 20, 25, 30])
        desired_ticks_hz = desired_ticks_hz[desired_ticks_hz <= max_hz]
        freq_tick_inds = [np.argmin(np.abs(visual_freqs_hz - hz)) for hz in desired_ticks_hz]

        # --- Panel 1 + 2: spectrograms ---
        for ci in range(2):
            ax = axes[ci]
            S = visual_output[ci]
            ax.imshow(S, aspect="auto", origin="lower", cmap="jet", vmin=-2, vmax=2)

            # Overlay bad epochs for that channel
            for e in np.where(bad_mask[ci])[0]:
                ax.axvspan(e, e + 1, color="magenta", alpha=0.45, lw=0)

            ax.set_ylabel(channel_names[ci])
            ax.set_yticks(freq_tick_inds)
            ax.set_yticklabels([f"{hz:g}" for hz in desired_ticks_hz])

        # --- Panel 3: correlation trace (diagnostic only) ---
        axc = axes[2]
        axc.plot(np.arange(n_epochs), eog_corr, lw=1.0)
        axc.set_ylabel("EOG corr")
        axc.set_ylim(-1.05, 1.05)

        axc.set_xticks(tick_positions)
        axc.set_xticklabels(tick_labels)
        axc.set_xlabel("Time (HH:MM)")

        plt.tight_layout()
        plt.show()

    return {
        "metric_names": ["eog_power", "hf_ratio", "eog_corr", "max_slope"],
        "metric_values": {
            "eog_power": eog_power,   # (2, epoch)
            "hf_ratio": hf_ratio,     # (2, epoch)
            "eog_corr": eog_corr,     # (epoch,)
            "max_slope": max_slope,   # (2, epoch)
        },
        "flat_mask": flat_mask,
        "saturation_mask": saturation_mask,
        "pop_mask": pop_mask,
        "hf_mask": hf_mask,
        "combined_flags": bad_mask,        # (2, epoch)
        "visual_output": visual_output,    # (2, freq, epoch)
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channels": channel_names,
            "visual_freqs_hz": freqs[plot_mask],
            "visual_epoch_times_s": (np.arange(n_epochs) * epoch_len).astype(float),
        }
    }


