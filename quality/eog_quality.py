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
    w0 = notch_hz / (fs / 2.0)
    if w0 <= 0 or w0 >= 1:
        return x
    b, a = iirnotch(w0=w0, Q=q)
    return filtfilt(b, a, x).astype(np.float32, copy=False)


def _eog_preprocess_for_view(x, fs,
                            bandpass=(0.3, 15.0),
                            notch_hz=None,
                            notch_q=30.0,
                            demean=True):
    """Typical PSG-style EOG visualization preprocessing.

    - Demean (removes DC)
    - Bandpass (default 0.3–15 Hz): emphasizes ocular deflections, removes drift and HF noise.
      (Some labs display up to 30–35 Hz; adjust bandpass if desired.)
    - Optional notch at 60 Hz (usually unnecessary if lowpass <= 15 Hz, but kept for flexibility).
    """
    y = x.astype(np.float32, copy=False)

    if demean:
        y = y - np.nanmean(y)

    b, a = _butter_bandpass(bandpass[0], bandpass[1], fs, order=4)
    y = filtfilt(b, a, np.nan_to_num(y)).astype(np.float32, copy=False)

    if notch_hz is not None:
        y = _apply_notch(y, fs, notch_hz=float(notch_hz), q=float(notch_q))

    return y


def calculate_eog_quality(signal, sampling_rate, channel_names=None,
                          epoch_len=30, plot=False,
                          # --- visualization / replay buffer controls ---
                          view_bandpass=(0.3, 15.0),
                          view_notch_hz=None,
                          plot_decimate_to_hz=200.0,
                          plot_seconds_per_screen=None):
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
        Contains masks, metrics, visual_output (2 x samples), and metadata.
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

    # EOG band for optional spectral checks (not required for staging; kept as a sanity metric)
    eog_band = (freqs >= 0.3) & (freqs <= 15.0)
    hf_band = (freqs > 15.0) & (freqs <= min(30.0, fs / 2.0))

    # -------------------------------
    # FFT / log-power (used for visualization output only)
    # -------------------------------
    F = np.abs(rfft(X, axis=2))  # (2, epoch, freq)


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
    # Visualization output (always computed): PSG-style preprocessed EOG traces
    # visual_output shape: (2, n_samples_trunc)
    # -------------------------------
    n_samples_trunc = n_epochs * epoch_samps
    X_cont = signal[:, :n_samples_trunc].astype(np.float32, copy=False)

    visual_output = np.zeros((2, n_samples_trunc), dtype=np.float32)
    for ci in range(2):
        visual_output[ci, :] = _eog_preprocess_for_view(
            X_cont[ci],
            fs=fs,
            bandpass=view_bandpass,
            notch_hz=view_notch_hz,
            notch_q=30.0,
            demean=True
        )

    # Hard checks so it cannot silently collapse/flatten inside this function
    assert visual_output.ndim == 2, visual_output.shape
    assert visual_output.shape == (2, n_samples_trunc), visual_output.shape

    visual_time_s = (np.arange(n_samples_trunc, dtype=np.float32) / fs)

    # -------------------------------
    # Optional plotting:

    if plot:

        # Two time-domain trace panels (one per EOG channel) + correlation diagnostic
        fig, axes = plt.subplots(3, 1, figsize=(18, 6), sharex=True)

        # Optional decimation for speed (visual only)
        if plot_decimate_to_hz is not None and plot_decimate_to_hz > 0 and fs > plot_decimate_to_hz:
            decim = int(np.floor(fs / plot_decimate_to_hz))
        else:
            decim = 1

        t_plot = visual_time_s[::decim]

        if plot_seconds_per_screen is not None:
            keep = t_plot <= float(plot_seconds_per_screen)
        else:
            keep = slice(None)

        # X-axis ticks every 30 minutes (labels HH:MM from start of record)
        tick_interval_epochs = max(1, int((30 * 60) // epoch_len))
        tick_positions_epochs = np.arange(0, n_epochs, tick_interval_epochs)
        tick_positions_s = tick_positions_epochs * epoch_len
        tick_labels = [time.strftime('%H:%M', time.gmtime(tsec)) for tsec in tick_positions_s]

        # --- Panel 1 + 2: time-domain EOG traces ---
        for ci in range(2):
            ax = axes[ci]
            y_plot = visual_output[ci, ::decim][keep]
            tp = t_plot[keep]
            ax.plot(tp, y_plot, lw=0.7)
            ax.set_ylabel(channel_names[ci])
            ax.grid(True, alpha=0.15)

            # Shade bad epochs for that channel (epoch-level flags)
            bad_epochs = np.where(bad_mask[ci])[0]
            for e in bad_epochs:
                start = e * epoch_len
                end = (e + 1) * epoch_len
                if plot_seconds_per_screen is not None and start > plot_seconds_per_screen:
                    break
                ax.axvspan(start, end, color="magenta", alpha=0.25, lw=0)

        # --- Panel 3: correlation trace (diagnostic only), aligned to time ---
        axc = axes[2]
        epoch_centers_s = (np.arange(n_epochs) + 0.5) * epoch_len
        axc.plot(epoch_centers_s, eog_corr, lw=1.0)
        axc.set_ylabel("EOG corr")
        axc.set_ylim(-1.05, 1.05)
        axc.grid(True, alpha=0.15)

        # Shared X formatting
        axes[-1].set_xticks(tick_positions_s)
        axes[-1].set_xticklabels(tick_labels)
        axes[-1].set_xlabel("Time (HH:MM)")

        if plot_seconds_per_screen is not None:
            axes[-1].set_xlim(0, float(plot_seconds_per_screen))

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
        "visual_output": visual_output,    # (2, n_samples_trunc)
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channels": channel_names,
            "visual_epoch_times_s": (np.arange(n_epochs) * epoch_len).astype(float),
            "visual_time_s": visual_time_s,
            "visual_output_shape": list(visual_output.shape),
        }
    }
