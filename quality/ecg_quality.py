import json
import numpy as np
from datetime import datetime, timezone, timedelta
from scipy.signal import butter, sosfiltfilt
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd  # only for input + optional conversion outside hot loop

# ---------- small helpers (NumPy-only in the loop) ----------
def clipping_ratio(seg, rail_min=-8333.0, rail_max=8333.0, near_pct=0.01):
    if seg.size == 0:
        return 1.0
    lower, upper = rail_min * (1 - near_pct), rail_max * (1 - near_pct)
    return float(np.mean((seg <= lower) | (seg >= upper)))

def flatline_ratio(seg, eps=1e-6):
    if seg.size < 2:
        return 1.0
    return float(np.mean(np.abs(np.diff(seg, prepend=seg[:1])) < eps))

def missing_ratio(n_present, n_expected):
    return float(max(0.0, 1.0 - n_present / max(1, n_expected)))

def ratio_summary(bad_n, total):
    good_n = total - bad_n
    return {
        "good_epochs": int(good_n),
        "bad_epochs": int(bad_n),
        "good_ratio": round(good_n / total, 3) if total else None,
        "bad_ratio":  round(bad_n / total, 3) if total else None,
    }

# ---------- main (drop-in) ----------
def run_ecg_qc(
    channel_name,
    channel_dataframes,     # dict of DataFrames; we extract once then go NumPy
    fs=200,
    epoch_len=30,
    thresholds=None,
    json_path=None,
    plot="overall"          # 'overall' | 'per-metric' | 'both' | 0
):
    # defaults
    th = {
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max":  0.50,
        "baseline_max": 0.15,
        "hr_min": 25.0,
        "hr_max": 220.0,
        "snr_min": 5.0,
    }
    if thresholds:
        th.update(thresholds)

    # --- extract arrays ONCE from pandas ---
    if channel_name not in channel_dataframes:
        raise KeyError(f"Channel '{channel_name}' not found.")
    df = channel_dataframes[channel_name]

    t_abs = pd.to_datetime(df["Absolute Time"], errors="coerce")
    if getattr(t_abs.dt, "tz", None) is None:
        t_abs = t_abs.dt.tz_localize("UTC")
    t_abs_ns = t_abs.view("int64")  # ns since epoch
    sig_np = pd.to_numeric(df[channel_name], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(sig_np) & np.isfinite(t_abs_ns)
    sig = sig_np[mask].astype(np.float32, copy=False)
    t_abs_ns = t_abs_ns[mask]
    if sig.size == 0:
        empty_per_metric = {k: ratio_summary(0, 0) for k in
                            ["Clipping","Flatline","Missing","Baseline","HR_Mean","SNR_dB"]}
        return [], empty_per_metric, {"total_epochs":0,"good_epochs":0,"bad_epochs":0,"good_ratio":None,"bad_ratio":None}

    # seconds from t0 for fast math; also keep datetime0 for plotting/ISO
    t0_ns = t_abs_ns[0]
    t_sec = (t_abs_ns - t0_ns) / 1e9
    t0_dt = datetime.fromtimestamp(t0_ns/1e9, tz=timezone.utc)

    # --- precompute globally (big speedups) ---
    spp = int(fs * epoch_len)
    n = sig.size

    sos_bp = butter(4, (0.25, 25), btype="bandpass", fs=fs, output="sos")
    sos_lp = butter(2, 0.30, btype="lowpass",  fs=fs, output="sos")

    ecg_bp    = sosfiltfilt(sos_bp, sig)
    ecg_clean = nk.ecg_clean(ecg_bp, sampling_rate=fs)

    sig_var   = float(np.var(ecg_bp))
    noise_var = float(np.var(ecg_bp - ecg_clean))
    snr_global = 10*np.log10(sig_var/(noise_var+1e-12)) if sig_var > 0 else np.nan

    # R-peaks once; slice per epoch by time window
    _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="engzeemod2012")
    r_idx = rpeaks.get("ECG_R_Peaks", np.array([], dtype=int))
    r_times = t_sec[r_idx] if r_idx.size else np.empty((0,), dtype=np.float32)

    # low-pass version for baseline-variance ratio
    ecg_lp = sosfiltfilt(sos_lp, sig)

    # masks for fast per-epoch ratios
    clip_mask = (sig <= -8333.0*0.99) | (sig >= 8333.0*0.99)  # adjust rails if needed
    flat_mask = np.abs(np.diff(sig, prepend=sig[:1])) < 1e-6

    # --- epoch loop (NumPy-only) ---
    starts = np.arange(0, n, spp, dtype=int)
    ends   = np.minimum(starts + spp, n)

    qc_rows = []
    for i, (s, e) in enumerate(zip(starts, ends), start=1):
        seg_bp = ecg_bp[s:e]
        seg_lp = ecg_lp[s:e]

        miss = missing_ratio(e - s, spp)
        clip = float(np.mean(clip_mask[s:e])) if e > s else 1.0
        flat = float(np.mean(flat_mask[s:e])) if e > s else 1.0

        total_var = float(np.var(seg_bp)) if seg_bp.size else 0.0
        base_var  = float(np.var(seg_lp)) if seg_lp.size else np.nan
        base_ratio = (base_var/total_var) if (total_var>0 and not np.isnan(base_var)) else np.nan

        # HR per epoch via R-peak time slicing
        t_s, t_e = t_sec[s], t_sec[e-1]
        L = np.searchsorted(r_times, t_s, side="left")
        R = np.searchsorted(r_times, t_e, side="left")
        r_slice = r_times[L:R]
        if r_slice.size >= 2:
            d_rr = np.diff(r_slice)
            hr = 60.0 / d_rr
            if hr.size > 1:
                m, sd = np.nanmean(hr), np.nanstd(hr) + 1e-12
                hr = hr[np.abs((hr - m)/sd) <= 6.0]
            hr_mean = float(np.nanmean(hr)) if hr.size else np.nan
        else:
            hr_mean = np.nan

        snr_db = snr_global

        # Make JSON-safe native booleans
        bad_clip = bool(clip > th["clipping_max"])
        bad_flat = bool(flat > th["flatline_max"])
        bad_miss = bool(miss > th["missing_max"])
        bad_base = bool((not np.isnan(base_ratio)) and (base_ratio > th["baseline_max"]))
        bad_hr   = bool((np.isnan(hr_mean)) or (hr_mean < th["hr_min"]) or (hr_mean > th["hr_max"]))
        bad_snr  = bool((np.isnan(snr_db))  or (snr_db < th["snr_min"]))
        bad_epoch = bool(bad_clip or bad_flat or bad_miss or bad_base or bad_hr or bad_snr)

        qc_rows.append({
            "Epoch": int(i),
            "Start_Time_ISO": (t0_dt + timedelta(seconds=float(t_sec[s]))).isoformat(),
            "End_Time_ISO":   (t0_dt + timedelta(seconds=float(t_sec[e-1]))).isoformat(),
            "Clipping_Ratio": float(clip),
            "Flatline_Ratio": float(flat),
            "Missing_Ratio":  float(miss),
            "Baseline_Wander_Ratio": None if np.isnan(base_ratio) else float(base_ratio),
            "HR_Mean": None if np.isnan(hr_mean) else float(hr_mean),
            "SNR_dB":  None if np.isnan(snr_db)  else float(snr_db),
            "Bad_Epoch": bad_epoch,
            "Bad_Clip": bad_clip,
            "Bad_Flatline": bad_flat,
            "Bad_Missing": bad_miss,
            "Bad_Baseline": bad_base,
            "Bad_HR": bad_hr,
            "Bad_SNR": bad_snr
        })

    # --- summaries ---
    total = len(qc_rows)
    def sum_flag(k): return sum(1 for r in qc_rows if r[k])

    per_metric_json = {
        "Clipping": ratio_summary(sum_flag("Bad_Clip"), total),
        "Flatline": ratio_summary(sum_flag("Bad_Flatline"), total),
        "Missing":  ratio_summary(sum_flag("Bad_Missing"), total),
        "Baseline": ratio_summary(sum_flag("Bad_Baseline"), total),
        "HR_Mean":  ratio_summary(sum_flag("Bad_HR"), total),
        "SNR_dB":   ratio_summary(sum_flag("Bad_SNR"), total),
    }
    overall_bad = sum_flag("Bad_Epoch")
    overall_json = {
        "total_epochs": int(total),
        "good_epochs": int(total - overall_bad),
        "bad_epochs": int(overall_bad),
        "good_ratio": round((total - overall_bad)/total, 3) if total else None,
        "bad_ratio":  round(overall_bad/total, 3) if total else None,
    }

    if json_path:
        with open(json_path, "w") as f:
            json.dump(
                {"per_epoch": qc_rows, "per_metric": per_metric_json, "overall": overall_json},
                f, indent=2
            )

    # --- plotting (supports 'overall' | 'per-metric' | 'both') ---
    if plot in ("overall", "both", "per-metric"):
        # build plot time array once (no pandas)
        times = [t0_dt + timedelta(seconds=float(s)) for s in t_sec]
        # overall
        if plot in ("overall", "both"):
            fig, ax = plt.subplots(figsize=(14,5))
            step = max(1, len(sig)//20000)
            ax.plot(times[::step], sig[::step], lw=0.8, color="black")
            for r in qc_rows:
                st = datetime.fromisoformat(r["Start_Time_ISO"])
                et = datetime.fromisoformat(r["End_Time_ISO"])
                ax.axvspan(st, et, color=("red" if r["Bad_Epoch"] else "green"), alpha=0.18)
            ax.set_title(f"{channel_name} — Overall QC (Red=Bad, Green=Good)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.grid(True); plt.tight_layout(); plt.show()

        # per-metric
        if plot in ("per-metric", "both"):
            metric_flag_map = {
                "Clipping_Ratio": "Bad_Clip",
                "Flatline_Ratio": "Bad_Flatline",
                "Missing_Ratio": "Bad_Missing",
                "Baseline_Wander_Ratio": "Bad_Baseline",
                "HR_Mean": "Bad_HR",
                "SNR_dB": "Bad_SNR",
            }
            for metric, flag in metric_flag_map.items():
                fig, ax = plt.subplots(figsize=(14,5))
                step = max(1, len(sig)//20000)
                ax.plot(times[::step], sig[::step], lw=0.8, color="black")
                for r in qc_rows:
                    st = datetime.fromisoformat(r["Start_Time_ISO"])
                    et = datetime.fromisoformat(r["End_Time_ISO"])
                    ax.axvspan(st, et, color=("red" if r[flag] else "green"), alpha=0.18)
                ax.set_title(f"{channel_name} — {metric}: QC (Red=Bad, Green=Good)")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax.grid(True); plt.tight_layout(); plt.show()

    # Return same 3-tuple your call expects
    return qc_rows, per_metric_json, overall_json