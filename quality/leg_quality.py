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


def calculate_leg_quality(
    channel_name,
    channel_dataframes,
    fs,
    epoch_len=30,
    clipping_max=0.50,
    flatline_max=0.50,
    missing_max=0.50,
    plot=False,
):
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found.")

    df = channel_dataframes[channel_name]

    # ============================================================
    # TIME + SIGNAL INGEST (NUMPY-SAFE)
    # ============================================================

    time = pd.to_datetime(df["Absolute Time"], errors="coerce", utc=True)
    time = time.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")

    sig = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(sig) & np.isfinite(time.astype("int64"))
    sig = sig[mask]
    time = time[mask]

    if sig.size == 0:
        return {
            "metric_names": ["clipping", "flatline", "missing"],
            "metric_values": {
                "clipping": np.array([]),
                "flatline": np.array([]),
                "missing": np.array([]),
            },
            "bad_masks": {
                "clipping": np.array([], dtype=bool),
                "flatline": np.array([], dtype=bool),
                "missing": np.array([], dtype=bool),
            },
            "combined_mask": np.array([], dtype=bool),
            "metadata": {"fs": fs, "epoch_len": epoch_len, "channel": channel_name},
        }

    # ============================================================
    # SETUP
    # ============================================================

    spp = int(fs * epoch_len)
    n_epochs = int(np.ceil(len(sig) / spp))

    clipping_vals = np.full(n_epochs, np.nan)
    flat_vals = np.full(n_epochs, np.nan)
    missing_vals = np.full(n_epochs, np.nan)
    epoch_start_times = np.empty(n_epochs, dtype="datetime64[ns]")

    # Local refs (micro-optimization)
    clip_fn = check_clipping_leg
    flat_fn = flatline_ratio_leg
    miss_fn = missing_ratio

    # ============================================================
    # EPOCH LOOP (LOGIC UNCHANGED)
    # ============================================================

    for i in range(n_epochs):
        s = i * spp
        e = min(s + spp, sig.size)

        seg = sig[s:e]

        epoch_start_times[i] = time[s]
        clipping_vals[i] = clip_fn(seg)
        flat_vals[i] = flat_fn(seg)
        missing_vals[i] = miss_fn(seg.size, spp)

    # ============================================================
    # MASKS
    # ============================================================

    masks = {
        "clipping": clipping_vals > clipping_max,
        "flatline": flat_vals > flatline_max,
        "missing": missing_vals > missing_max,
    }

    combined_mask = np.any(np.column_stack(list(masks.values())), axis=1)

    # ============================================================
    # PLOTTING (UNCHANGED)
    # ============================================================

    if plot:
        def shade_epochs(ax, mask_array):
            for i, bad in enumerate(mask_array):
                st = epoch_start_times[i]
                et = st + np.timedelta64(epoch_len, "s")
                ax.axvspan(st, et, color=("red" if bad else "green"), alpha=0.18)

        metric_names = list(masks.keys())
        n_rows = 1 + len(metric_names)

        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(16, 1.5 * n_rows),
            sharex=True
        )

        axes[0].plot(time, sig, lw=0.8, color="black")
        shade_epochs(axes[0], combined_mask)
        axes[0].set_title(f"{channel_name} — Overall LEG QC")
        axes[0].grid(True)

        for i, name in enumerate(metric_names, start=1):
            axes[i].plot(time, sig, lw=0.8, color="black")
            shade_epochs(axes[i], masks[name])
            axes[i].set_title(name.upper())
            axes[i].grid(True)

        xmin = time[0]
        xmax = time[-1]

        for ax in axes:
            ax.set_xlim(xmin, xmax)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    # ============================================================
    # RETURN
    # ============================================================

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