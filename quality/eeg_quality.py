import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter

def calculate_quality(signal, sampling_rate, channel_names=None,
                      epoch_len=30, ar_thresh=6, chan_range=None, **kwargs):
    """
    EEG artifact quality check for Physio-QC-Toolkit.
    
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
    chan_range : list of int, optional
        Channels to include (defaults to all)
    
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
    if chan_range is None:
        chan_range = range(signal.shape[0])

    fs = sampling_rate
    epoch_samps = int(fs * epoch_len)
    n_epochs = signal.shape[1] // epoch_samps
    freqs = rfftfreq(epoch_samps, 1/fs)

    artifact_scores = np.zeros((len(chan_range), n_epochs))
    flags = np.zeros_like(artifact_scores, dtype=bool)

    for ci, ch in enumerate(chan_range):
        sig = signal[ci, :n_epochs*epoch_samps].reshape(n_epochs, epoch_samps)
        for ei, ep in enumerate(sig):
            F = np.abs(np.fft.fft(ep))[1:500]
            F = np.log10(F / np.mean(F))
            art_val = (np.mean(F[0:20]) / np.abs(np.mean(F[20:50]))) / np.abs(np.mean(F[400:500]))
            artifact_scores[ci, ei] = art_val
            if art_val > ar_thresh:
                flags[ci, ei] = True

    ch_names = [channel_names[i] for i in chan_range] if channel_names else [f"ch{i}" for i in chan_range]
    
    return {
        "metric_names": ["artifact_score"],
        "metric_values": artifact_scores,
        "flags": flags,
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channels": ch_names
        }
    }
