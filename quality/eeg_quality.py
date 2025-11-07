import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter
import time

def calculate_quality(signal, sampling_rate, channel_names=None,
                      epoch_len=30, ar_thresh=6, plot=False, **kwargs):
    """
    EEG artifact and flatline quality check.

    Performs two main quality checks:
    1. Spectral noise detection using a triple-ratio metric comparing low-, mid-, and high-frequency bands.
    2. Flatline detection using variance, peak-to-peak amplitude, and repeated-value ratios.

    Parameters
    ----------
    signal : np.ndarray
        EEG array of shape (n_channels, n_samples)
    sampling_rate : int
        Sampling rate in Hz
    channel_names : list of str, optional
        Channel labels (used in metadata)
    epoch_len : float
        Epoch length in seconds
    ar_thresh : float
        Artifact rejection threshold
    plot : bool
        Whether to run zscoring, smoothing, and plotting

    Returns
    -------
    dict
        {
          "metric_names": ["artifact_score"],
          "metric_values": np.ndarray (n_channels, n_epochs),
          "flags": np.ndarray (n_channels, n_epochs),
          "metadata": {...}
        }
    """

    fs = sampling_rate  # sampling frequency
    epoch_samps = int(fs * epoch_len)  # number of samples per epoch
    n_ch, n_samples = signal.shape
    n_epochs = n_samples // epoch_samps  # total number of epochs

    # --- FFT and frequency setup ---
    freqs = rfftfreq(epoch_samps, 1/fs)  # compute frequency bins for rFFT
    ref_mask = (freqs >= 1) & (freqs <= 40)  # reference band for normalization
    ref_idx = np.where(ref_mask)[0]
    plot_mask = (freqs <= 30)  # limit plotted frequency range

    # --- Reshape signal into epochs ---
    X = signal[:, :n_epochs * epoch_samps].astype(np.float32, copy=False)
    X = X.reshape(n_ch, n_epochs, epoch_samps)

    # --- FFT computation (normalized per channel) ---
    F = np.abs(rfft(X, axis=2))  # magnitude of FFT
    F_mean = F[..., ref_idx].mean(axis=2, keepdims=True)  # average power in reference band
    F /= F_mean  # normalize spectrum per epoch
    logpow = np.log10(F + 1e-8)  # convert to log power scale

    # --- Artifact detection via triple spectral ratio ---
    # Ratio of low (0–20 Hz) to mid (20–50 Hz) to high (400–500 Hz) spectral bands
    F2 = F[..., 1:500]
    F2 /= F2.mean(axis=2, keepdims=True)
    F2 = np.log10(F2 + 1e-8)

    art_val = ((np.mean(F2[:, :, 0:20], 2) /
                np.abs(np.mean(F2[:, :, 20:50], 2))) /
               np.abs(np.mean(F2[:, :, 400:500], 2)))

    noise_mask = art_val > ar_thresh  # mark noisy epochs exceeding threshold

    # --- Flatline detection ---
    # Detects abnormally low variance and amplitude, or excessive repeated samples
    epoch_var = np.var(X, axis=2)
    epoch_ptp = np.ptp(X, axis=2)
    diffs = np.diff(X, axis=2)
    repeat_ratio = np.mean(np.abs(diffs) < 1e-6, axis=2)  # proportion of nearly identical samples

    # Determine dynamic thresholds (relative to lowest 5th percentile)
    var_thresh = np.percentile(epoch_var, 5) * 0.2
    amp_thresh = np.percentile(epoch_ptp, 5) * 0.2

    flat_mask = ((epoch_var < var_thresh) & (epoch_ptp < amp_thresh)) | (repeat_ratio > 0.98)

    # --- Combine artifact flags ---
    art_mask = noise_mask | flat_mask  # unified mask for all detected artifacts

    # --- Optional Z-score normalization and visualization ---
    if plot:
        # Compute z-score using only valid (non-flatline) epochs
        valid_mask = ~flat_mask
        valid_flat = logpow[valid_mask]
        mean = np.nanmean(valid_flat, axis=0, keepdims=True)
        std = np.nanstd(valid_flat, axis=0, keepdims=True)
        logpow_z = (logpow - mean) / (std + 1e-8)
        logpow_z[flat_mask] = -5  # mark flat epochs visually

        # Apply light Gaussian smoothing for visualization
        smooth = gaussian_filter(logpow_z, sigma=(0, 1, 1))[:, :, plot_mask]
        smooth = np.transpose(smooth, (0, 2, 1))  # reorder to (ch, freq, epoch)

        # --- Plotting setup ---
        fig, axes = plt.subplots(n_ch, 1, figsize=(18, 1.5 * n_ch), sharex=True)
        if n_ch == 1:
            axes = [axes]

        # Define time tick positions (every 30 minutes)
        tick_interval_sec = 30 * 60
        tick_interval_epochs = tick_interval_sec // epoch_len
        tick_positions = np.arange(0, n_epochs, tick_interval_epochs)
        tick_labels = [time.strftime('%H:%M', time.gmtime(t * epoch_len)) for t in tick_positions]

        # --- Channel plots ---
        for ci, ch in enumerate(channel_names or [f"ch{ci}" for ci in range(n_ch)]):
            ax = axes[ci]
            S, n_freqs = smooth[ci], smooth.shape[1]
            # Plot log-power spectrogram
            ax.imshow(S, aspect="auto", origin="lower", cmap="jet", vmin=-1, vmax=1,
                      extent=[0, n_epochs, 0, n_freqs])
            # Highlight detected artifacts
            for e in np.where(art_mask[ci])[0]:
                ax.axvspan(e, e + 1, color='magenta', alpha=0.45, lw=0)
            ax.set_ylabel(ch, fontsize=12)
            ax.set_ylim(0, n_freqs)
            freq_ticks = np.linspace(0, n_freqs-1, 4)
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(['0', '10', '20', '30'], fontsize=11)
            if ci == n_ch - 1:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=10)
                ax.set_xlabel("Time (HH:MM)", fontsize=14)

        # Common axis labels and layout adjustments
        fig.text(0.01, 0.5, "Frequency (Hz)", va="center", rotation="vertical", fontsize=16)
        plt.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.93, hspace=0.15)
        plt.tight_layout(rect=[0.02, 0, 1, 1])
        plt.show()

    # --- Return structured results ---
    ch_names = channel_names or [f"ch{i}" for i in range(n_ch)]
    return {
        "metric_names": ["noise_score"],
        "metric_values": art_val,  # numeric noise metric per epoch
        "flatline_mask": flat_mask,  # boolean mask per epoch
        "noise_mask": noise_mask,    # boolean mask per epoch
        "combined_flags": art_mask,  # union of noise + flatline
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channels": ch_names
        }
    }
