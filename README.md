#  physio-qc-toolkit

A lightweight **Python toolkit** for **physiological signal quality control (QC)** and visualization across multimodal PSG and wearable data (e.g., ECG, thermistor airflow, pressure, PPG).

---

## 🐍 Requirements

- **Python 3.11**
- Install dependencies via:
  ```bash
  pip install -r requirements.txt

##  Overview

This toolkit provides an **end-to-end pipeline** to:
1. **Read EDF files** and extract channel data into structured DataFrames  
2. **Run signal quality checks** (QC) on ECG, respiratory, and other biosignals  
3. **Compute per-epoch metrics** such as clipping, flatline ratio, missing data, baseline drift, HR, and SNR  
4. **Visualize quality flags** with color-coded (red/green) spans  
5. **Generate JSON summaries** for automated downstream analyses

---

##  Repository Structure

physio-qc-toolkit/
├── ecg/ # ECG-specific feature extraction (e.g., HR, HRV, SNR)
├── quality/ # QC pipelines and visualization functions
├── read/ # EDF file reading utilities
├── edf_signal_quality.ipynb # Main Jupyter notebook (run + visualize QC)
├── LICENSE
├── README.md
└── init.py

---

##  Features

- **EDF Reader**: Handles EDF headers, sampling rates, labels, and transducers  
- **Channel Disambiguation**: Creates unique labels when duplicates exist  
- **Epoch-Based QC**: 30s windows (configurable) with per-epoch stats  
- **Metrics:**
  - Clipping Ratio (`>50%` = bad)
  - Flatline Ratio (`>50%` = bad)
  - Missing Ratio (`>50%` = bad)
  - Baseline Wander (`>15%` power <0.3 Hz = bad)
  - Heart Rate Range (`25–220 bpm`)
  - SNR (`<5 dB` = bad)
- **Visualization:**  
  - Per-metric and overall QC plots (green = good, red = bad)
- **JSON Export:**  
  - Ratio of good/bad epochs per metric  

---

##  Example Usage

from read.read_edf import read_edf_to_dataframes
from quality.run_qc import run_ecg_qc

# Step 1: Read EDF file
channel_dataframes = read_edf_to_dataframes("ABC100110013333PSG06.edf")

# Step 2: Run QC on ECG II
qc_df, per_metric_json, overall_json = run_ecg_qc(
    "ECG II",
    channel_dataframes=channel_dataframes,
    fs=200,
    epoch_len=30,
    thresholds={
        "clipping_max": 0.50,
        "flatline_max": 0.50,
        "missing_max":  0.50,
        "baseline_max": 0.15,
        "hr_min": 25.0,
        "hr_max": 220.0,
        "snr_min": 5.0,
    },
    json_path="qc_summary.json",
    plot="per-metric"
)

# Step 3: Print summary
print(overall_json)

 Output Example

=== Overall QC Summary ===
{'total_epochs': 180, 'good_epochs': 157, 'bad_epochs': 23, 'good_ratio': 0.872, 'bad_ratio': 0.128}

===  Per-metric === 
{
  "Clipping": {"good_ratio": 0.98, "bad_ratio": 0.02},
  "Flatline": {"good_ratio": 1.00, "bad_ratio": 0.00},
  "Missing": {"good_ratio": 0.97, "bad_ratio": 0.03},
  "Baseline": {"good_ratio": 0.85, "bad_ratio": 0.15},
  "HR_Mean": {"good_ratio": 0.90, "bad_ratio": 0.10},
  "SNR_dB": {"good_ratio": 0.95, "bad_ratio": 0.05}
}


##  Visualization

Each 30-second epoch is shaded according to QC results:
- 🟢 **Green** → passes all quality checks  
- 🔴 **Red** → fails at least one metric threshold  

### Example: SNR Quality Check
![ECG II — SNR_dB Quality Check](SNR.png)

License
This project is licensed under the MIT License — see the LICENSE