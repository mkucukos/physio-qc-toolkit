import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

# ----------------------------------------------------
# Helper metrics (unchanged)
# ----------------------------------------------------

def check_clipping_leg(signal, pmin, pmax, edge_pct=0.1):
    if signal.size == 0:
        return np.nan
    lower = pmin + edge_pct * (pmax - pmin)
    upper = pmax - edge_pct * (pmax - pmin)
    clipped = (signal <= lower) | (signal >= upper)
    return float(np.mean(clipped))


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

def run_leg_qc(
    channel_name,
    channel_dataframes,
    fs,
    pmin,
    pmax,
    epoch_len=30,
    json_path=None,
    plot="per-metric",
    clipping_max=0.50,
    flatline_max=0.50,
    missing_max=0.50,
):
    """
    LEG QC using df['Absolute Time'] directly (no timezone logic).
    Flow-style outputs:
      per_epoch, per_metric_json, overall_json
    """

    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found.")

    df = channel_dataframes[channel_name]

    # --- extract time + signal exactly like your plotting ---
    time = pd.to_datetime(df["Absolute Time"], errors="coerce")
    sig = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(sig) & time.notna()
    sig = sig[mask]
    time = time.loc[mask].reset_index(drop=True)


    if sig.size == 0:
        empty = {
            "total_epochs": 0,
            "good_epochs": 0,
            "bad_epochs": 0,
            "good_ratio": None,
            "bad_ratio": None,
        }
        return [], {}, empty

    spp = int(fs * epoch_len)
    starts = np.arange(0, len(sig), spp)
    ends = np.minimum(starts + spp, len(sig))

    per_epoch = []

    # ------------------------------------------------
    # Epoch loop
    # ------------------------------------------------
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        seg = sig[s:e]

        clip = check_clipping_leg(seg, pmin, pmax)
        flat = flatline_ratio_leg(seg)
        miss = missing_ratio(len(seg), spp)

        bad_clip = bool(np.isfinite(clip) and clip > clipping_max)
        bad_flat = bool(np.isfinite(flat) and flat > flatline_max)
        bad_miss = bool(np.isfinite(miss) and miss > missing_max)

        bad_epoch = bad_clip or bad_flat or bad_miss

        per_epoch.append(
            {
                "Epoch": i,
                "Start_Time": time.iloc[s],
                "End_Time": time.iloc[e - 1],
                "Clipping_Ratio": clip,
                "Flatline_Ratio": flat,
                "Missing_Ratio": miss,
                "Bad_Epoch": bad_epoch,
                "Bad_Clip": bad_clip,
                "Bad_Flatline": bad_flat,
                "Bad_Missing": bad_miss,
                "Raw_Data": seg.tolist(),
            }
        )

    # ------------------------------------------------
    # Summaries
    # ------------------------------------------------
    total = len(per_epoch)

    def count(flag):
        return sum(r[flag] for r in per_epoch)

    per_metric_json = {
        "Clipping": ratio_summary(count("Bad_Clip"), total),
        "Flatline": ratio_summary(count("Bad_Flatline"), total),
        "Missing": ratio_summary(count("Bad_Missing"), total),
    }

    overall_bad = count("Bad_Epoch")
    overall_json = {
        "total_epochs": total,
        "good_epochs": total - overall_bad,
        "bad_epochs": overall_bad,
        "good_ratio": round((total - overall_bad) / total, 3) if total else None,
        "bad_ratio": round(overall_bad / total, 3) if total else None,
    }

    # ------------------------------------------------
    # Plotting (same style as your raw plots)
    # ------------------------------------------------
    if plot in ("overall", "per-metric", "both"):

        def shade(ax, flag):
            for r in per_epoch:
                ax.axvspan(
                    r["Start_Time"],
                    r["End_Time"],
                    color=("red" if r[flag] else "green"),
                    alpha=0.18,
                )

        step = 1

        if plot in ("overall", "both"):
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(time.iloc[::step], sig[::step], lw=0.8, color="black")
            shade(ax, "Bad_Epoch")
            ax.set_title(f"{channel_name} — Overall LEG QC")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True)
            plt.tight_layout()
            plt.show()

        if plot in ("per-metric", "both"):
            for metric, flag in {
                "Clipping": "Bad_Clip",
                "Flatline": "Bad_Flatline",
                "Missing": "Bad_Missing",
            }.items():
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(time.iloc[::step], sig[::step], lw=0.8, color="black")
                shade(ax, flag)
                ax.set_title(f"{channel_name} — {metric} QC")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax.grid(True)
                plt.tight_layout()
                plt.show()

    # --- optional JSON save ---
    if json_path:
        with open(json_path, "w") as f:
            json.dump(
                {
                    "per_epoch": per_epoch,
                    "per_metric": per_metric_json,
                    "overall": overall_json,
                },
                f,
                indent=2,
            )

    return per_epoch, per_metric_json, overall_json
