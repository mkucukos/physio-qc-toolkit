import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def run_clipping_qc(
    channel_name: str,
    channel_dataframes: dict,
    fs: int,
    epoch_len: int = 30,
    rail_min: float = None,         # e.g., 0.0 for Technical
    rail_max: float = None,         # e.g., 65535.0 for Technical
    near_pct: float = 0.01,         # how close to the rails to count as "clipped"
    plot: bool = False              # optional quick plot (matplotlib)
):
    """
    Compute ONLY the clipping ratio per epoch for a given channel.

    Clipping ratio = fraction of samples that lie within 'near_pct' of the rail_min or rail_max.

    Parameters
    ----------
    channel_name : str
        Name of the channel (must be a key in channel_dataframes).
    channel_dataframes : dict[str, DataFrame]
        Each DataFrame must have columns ["Absolute Time", channel_name].
        Optionally, DataFrame.attrs may include 'physical_min' and 'physical_max'.
    fs : int
        Sampling frequency (Hz) for this channel.
    epoch_len : int
        Epoch length in seconds.
    rail_min, rail_max : float, optional
        Rails to test clipping against. If None, tries df.attrs['physical_min/max'].
        If still None, falls back to the observed min/max of the signal.
    near_pct : float
        Percentage margin near each rail considered as "clip" (e.g., 0.01 = within 1% of rail).
    plot : bool
        If True, plots the signal and shades epochs by clipping severity.

    Returns
    -------
    quality_df : DataFrame
        Columns: [Epoch, Start_Time, End_Time, Clipping_Ratio]
    per_metric : dict
        {'Clipping': {'good_epochs', 'bad_epochs', 'good_ratio', 'bad_ratio'}}
        (bad = clipping_ratio > 0.50 by default threshold for reporting)
    overall_json : dict
        {'total_epochs', 'good_epochs', 'bad_epochs', 'good_ratio', 'bad_ratio'}
    """
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found.")

    df = channel_dataframes[channel_name].copy()

    # --- time & sort ---
    df["Absolute Time"] = pd.to_datetime(df["Absolute Time"], errors="coerce")
    if getattr(df["Absolute Time"].dt, "tz", None) is None:
        df["Absolute Time"] = df["Absolute Time"].dt.tz_localize("UTC")
    df = df.dropna(subset=["Absolute Time", channel_name]).sort_values("Absolute Time")

    # --- series ---
    sig = pd.to_numeric(df[channel_name], errors="coerce")
    mask = ~sig.isna()
    df = df.loc[mask]
    sig = sig.loc[mask].to_numpy()

    # --- determine rails ---
    rmin = rail_min
    rmax = rail_max
    if (rmin is None) or (rmax is None):
        # try DataFrame attrs first
        if hasattr(channel_dataframes[channel_name], "attrs"):
            rmin = channel_dataframes[channel_name].attrs.get("physical_min", rmin)
            rmax = channel_dataframes[channel_name].attrs.get("physical_max", rmax)
    if (rmin is None) or (rmax is None):
        # fallback to observed signal range
        rmin = float(np.nanmin(sig))
        rmax = float(np.nanmax(sig))

    # --- clipping function (ONLY metric we compute) ---
    low_bound  = rmin + (rmax - rmin) * (near_pct)
    high_bound = rmax - (rmax - rmin) * (near_pct)

    def clipping_ratio_epoch(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        # Near rails means <= low_bound OR >= high_bound
        clipped = np.logical_or(x <= low_bound, x >= high_bound)
        return float(np.mean(clipped))

    # --- epoching ---
    samples_per_epoch = int(fs * epoch_len)
    n_epochs = int(np.ceil(len(sig) / samples_per_epoch)) if samples_per_epoch > 0 else 0
    t0 = df["Absolute Time"].iloc[0]

    rows = []
    for i in range(n_epochs):
        s = i * samples_per_epoch
        e = min((i + 1) * samples_per_epoch, len(sig))
        if s >= e:
            continue
        epoch_sig = sig[s:e]
        clip = clipping_ratio_epoch(epoch_sig)
        rows.append({
            "Epoch": i + 1,
            "Start_Time": df["Absolute Time"].iloc[s],
            "End_Time": df["Absolute Time"].iloc[e - 1],
            "Clipping_Ratio": clip
        })

    quality_df = pd.DataFrame(rows)

    # --- simple per-metric / overall summaries (bad if > 0.50; tweak if you like) ---
    total = len(quality_df)
    if total > 0:
        bad_mask = quality_df["Clipping_Ratio"] > 0.50
        bad_n = int(bad_mask.sum())
    else:
        bad_n = 0

    per_metric = {
        "Clipping": {
            "good_epochs": total - bad_n,
            "bad_epochs": bad_n,
            "good_ratio": round((total - bad_n) / total, 3) if total else np.nan,
            "bad_ratio": round(bad_n / total, 3) if total else np.nan,
        }
    }
    overall_json = {
        "total_epochs": total,
        "good_epochs": total - bad_n,
        "bad_epochs": bad_n,
        "good_ratio": round((total - bad_n) / total, 3) if total else np.nan,
        "bad_ratio": round(bad_n / total, 3) if total else np.nan,
    }

    # --- optional plot ---
    if plot and total > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df["Absolute Time"], pd.to_numeric(df[channel_name], errors="coerce"), lw=0.6, color="black")
        ax.set_title(f"{channel_name} — Clipping-only QC (red shade = high clipping)")
        ax.set_ylabel("Signal")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.grid(True, alpha=0.3)

        for _, r in quality_df.iterrows():
            color = "red" if r["Clipping_Ratio"] > 0.50 else "green"
            ax.axvspan(r["Start_Time"], r["End_Time"], color=color, alpha=0.15)

        plt.tight_layout()
        plt.show()

    return quality_df, per_metric, overall_json
