import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import json
from datetime import datetime, timezone, timedelta

def run_clipping_qc(
    channel_name: str,
    channel_dataframes: dict,
    fs: int,
    epoch_len: int = 30,
    rail_min: float = None,         # e.g., 0.0 for Technical
    rail_max: float = None,         # e.g., 65535.0 for Technical
    near_pct: float = 0.01,         # how close to the rails to count as "clipped"
    plot: bool = False,             # optional quick plot (matplotlib)
    json_path: str | None = None,   # optional: write a single JSON file
):
    """
    Compute ONLY the clipping ratio per epoch for a given channel, JSON-friendly.

    Returns
    -------
    per_epoch : list[dict]
        [{"Epoch", "Start_Time_ISO", "End_Time_ISO", "Clipping_Ratio", "Bad_Clip"}, ...]
    per_metric_json : dict
        {"Clipping": {"good_epochs","bad_epochs","good_ratio","bad_ratio"}}
        (bad threshold fixed at 0.50; adjust here if you like)
    overall_json : dict
        {"total_epochs","good_epochs","bad_epochs","good_ratio","bad_ratio"}
    """
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found.")

    df = channel_dataframes[channel_name]

    # --- time & sanitize (pandas once) ---
    t_abs = pd.to_datetime(df["Absolute Time"], errors="coerce")
    if getattr(t_abs.dt, "tz", None) is None:
        t_abs = t_abs.dt.tz_localize("UTC")
    t_abs_ns = t_abs.astype("int64", copy=False)  # ns since epoch

    sig_np = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(sig_np) & np.isfinite(t_abs_ns)
    sig = sig_np[mask].astype(np.float32, copy=False)
    t_abs_ns = t_abs_ns[mask]

    if sig.size == 0:
        empty_per_epoch = []
        empty_metric = {"Clipping": {"good_epochs": 0, "bad_epochs": 0, "good_ratio": None, "bad_ratio": None}}
        empty_overall = {"total_epochs": 0, "good_epochs": 0, "bad_epochs": 0, "good_ratio": None, "bad_ratio": None}
        if json_path:
            with open(json_path, "w") as f:
                json.dump({"per_epoch": empty_per_epoch, "per_metric": empty_metric, "overall": empty_overall}, f, indent=2)
        return empty_per_epoch, empty_metric, empty_overall

    # --- rails (attrs → fallback to observed range) ---
    rmin = rail_min
    rmax = rail_max
    if (rmin is None) or (rmax is None):
        attrs = getattr(channel_dataframes[channel_name], "attrs", {})
        rmin = attrs.get("physical_min", rmin)
        rmax = attrs.get("physical_max", rmax)
    if (rmin is None) or (rmax is None):
        rmin = float(np.nanmin(sig))
        rmax = float(np.nanmax(sig))

    # bounds within which we call it "near rails"
    low_bound  = rmin + (rmax - rmin) * (near_pct)
    high_bound = rmax - (rmax - rmin) * (near_pct)

    # --- numpy time base ---
    t0_ns = t_abs_ns[0]
    t_sec = (t_abs_ns - t0_ns) / 1e9
    t0_dt = datetime.fromtimestamp(t0_ns / 1e9, tz=timezone.utc)

    # --- epoching ---
    spp = int(fs * epoch_len)
    n = sig.size
    starts = np.arange(0, n, spp, dtype=int)
    ends   = np.minimum(starts + spp, n)

    # --- per-epoch (NumPy only) ---
    per_epoch = []
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        if s >= e:
            continue
        seg = sig[s:e]
        if seg.size == 0:
            clip = np.nan
        else:
            clipped = (seg <= low_bound) | (seg >= high_bound)
            clip = float(np.mean(clipped))

        bad_clip = bool(np.isfinite(clip) and (clip > 0.50))
        per_epoch.append({
            "Epoch": int(i),
            "Start_Time_ISO": (t0_dt + timedelta(seconds=float(t_sec[s]))).isoformat(),
            "End_Time_ISO":   (t0_dt + timedelta(seconds=float(t_sec[e-1]))).isoformat(),
            "Clipping_Ratio": None if not np.isfinite(clip) else float(clip),
            "Bad_Clip": bad_clip,
        })

    # --- summaries (JSON-safe) ---
    total = len(per_epoch)
    bad_n = sum(1 for r in per_epoch if r["Bad_Clip"])
    good_n = total - bad_n
    per_metric_json = {
        "Clipping": {
            "good_epochs": int(good_n),
            "bad_epochs": int(bad_n),
            "good_ratio": round(good_n / total, 3) if total else None,
            "bad_ratio": round(bad_n / total, 3) if total else None,
        }
    }
    overall_json = {
        "total_epochs": int(total),
        "good_epochs": int(good_n),
        "bad_epochs": int(bad_n),
        "good_ratio": round(good_n / total, 3) if total else None,
        "bad_ratio": round(bad_n / total, 3) if total else None,
    }

    # --- optional plot ---
    if plot and total > 0:
        times = [t0_dt + timedelta(seconds=float(s)) for s in t_sec]
        fig, ax = plt.subplots(figsize=(14, 4))
        step = max(1, n // 20000)
        ax.plot(times[::step], sig[::step], lw=0.6, color="black")
        ax.set_title(f"{channel_name} — Clipping-only QC (red = bad, green = good)")
        ax.set_ylabel("Signal")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True, alpha=0.3)
        for r in per_epoch:
            st = datetime.fromisoformat(r["Start_Time_ISO"])
            et = datetime.fromisoformat(r["End_Time_ISO"])
            ax.axvspan(st, et, color=("red" if r["Bad_Clip"] else "green"), alpha=0.15)
        plt.tight_layout()
        plt.show()

    # --- optional save ---
    if json_path:
        with open(json_path, "w") as f:
            json.dump({"per_epoch": per_epoch, "per_metric": per_metric_json, "overall": overall_json}, f, indent=2)

    return per_epoch, per_metric_json, overall_json