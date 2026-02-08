import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

# ----------------------------------------------------
# Helper metrics (unchanged)
# ----------------------------------------------------

def check_clipping_leg(signal):
    """
    Detect LEG sensor saturation using adaptive plateau detection.
    Works without assuming fixed pmin/pmax scaling.
    Returns fraction of samples stuck near min/max rails.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size < 10:
        return np.nan

    s_min = np.nanmin(signal)
    s_max = np.nanmax(signal)
    s_rng = s_max - s_min

    if s_rng <= 1e-12:
        return 1.0  # completely flat / dead sensor

    # Samples within 0.1% of dynamic range = "railing"
    eps = s_rng * 0.001

    near_min = np.mean(signal <= (s_min + eps))
    near_max = np.mean(signal >= (s_max - eps))

    return float(max(near_min, near_max))


def flatline_ratio_leg(signal, eps=1e-6):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2:
        return 1.0

    epoch_var = np.var(signal)
    epoch_ptp = np.ptp(signal)
    repeat_ratio = np.mean(np.abs(np.diff(signal)) < eps)

    flat = ((epoch_var < 1e-6 and epoch_ptp < 1e-3) or (repeat_ratio > 0.98))
    return float(flat)


def missing_ratio(n_present, n_expected):
    if n_expected <= 0:
        return np.nan
    return float(max(0.0, 1.0 - n_present / n_expected))


def ratio_summary(bad_n, total):
    good_n = total - bad_n
    return {
        "good_epochs": int(good_n),
        "bad_epochs": int(bad_n),
        "good_ratio": round(good_n / total, 3) if total else None,
        "bad_ratio": round(bad_n / total, 3) if total else None,
    }

# ----------------------------------------------------
# LEG QC (time-aligned with raw plotting)
# ----------------------------------------------------

def calculate_leg_quality(
    channel_name,
    channel_dataframes,
    fs,
    epoch_len=30,
    clipping_max=0.50,
    flatline_max=0.50,
    missing_max=0.50,
    plot="overall",
):
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found.")

    df = channel_dataframes[channel_name]

    # --- Time & signal ---
    time = pd.to_datetime(df["Absolute Time"], errors="coerce")
    sig = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(sig) & time.notna()
    sig = sig[mask]
    time = time.loc[mask].reset_index(drop=True)

    if sig.size == 0:
        return {
            "metric_names": ["clipping", "flatline", "missing"],
            "metric_values": {},
            "bad_masks": {},
            "combined_mask": np.array([]),
            "metadata": {"fs": fs, "epoch_len": epoch_len, "channel": channel_name},
        }

    spp = int(fs * epoch_len)
    starts = np.arange(0, len(sig), spp)
    ends = np.minimum(starts + spp, len(sig))
    n_epochs = len(starts)

    # ---------------- Metric Arrays ----------------
    clipping_vals = np.zeros(n_epochs)
    flat_vals = np.zeros(n_epochs)
    missing_vals = np.zeros(n_epochs)
    epoch_start_times = []

    # ---------------- Epoch Loop ----------------
    for i, (s, e) in enumerate(zip(starts, ends)):
        seg = sig[s:e]
        epoch_start_times.append(time.iloc[s])

        clipping_vals[i] = check_clipping_leg(seg)
        flat_vals[i] = flatline_ratio_leg(seg)
        missing_vals[i] = missing_ratio(len(seg), spp)

    epoch_start_times = np.array(epoch_start_times)

    # ---------------- Boolean Masks ----------------
    masks = {
        "clipping": clipping_vals > clipping_max,
        "flatline": flat_vals > flatline_max,
        "missing": missing_vals > missing_max,
    }

    combined_mask = np.any(np.column_stack(list(masks.values())), axis=1)

    # ---------------- Plotting ----------------
    def shade_epochs(ax, mask_array):
        for i, bad in enumerate(mask_array):
            start = epoch_start_times[i]
            end = start + pd.Timedelta(seconds=epoch_len)
            ax.axvspan(start, end, color=("red" if bad else "green"), alpha=0.18)

    if plot in ("overall", "both"):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time, sig, lw=0.8, color="black")
        shade_epochs(ax, combined_mask)
        ax.set_title(f"{channel_name} — Overall LEG QC")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    if plot in ("per-metric", "both"):
        for name, mask in masks.items():
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(time, sig, lw=0.8, color="black")
            shade_epochs(ax, mask)
            ax.set_title(f"{channel_name} — {name.upper()} QC")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True)
            plt.tight_layout()
            plt.show()

    return {
        "metric_names": ["clipping", "flatline", "missing"],
        "metric_values": {
            "clipping": clipping_vals,
            "flatline": flat_vals,
            "missing": missing_vals,
        },
        "bad_masks": masks,
        "combined_mask": combined_mask,
        "metadata": {
            "fs": fs,
            "epoch_len": epoch_len,
            "channel": channel_name,
        },
    }