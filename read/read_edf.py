import pyedflib
import pandas as pd
import numpy as np
from datetime import timezone
from collections import defaultdict
from typing import Dict

def read_edf_metadata(file_path: str) -> pd.DataFrame:
    """
    Fast metadata read for all channels (vectorized where possible).
    Returns a pandas DataFrame. No printing.
    """
    f = None
    try:
        f = pyedflib.EdfReader(file_path)
        n = f.signals_in_file
        # Pre-fetch once to reduce Python overhead
        labels   = [f.getLabel(i)                 for i in range(n)]
        sfs      = [f.getSampleFrequency(i)       for i in range(n)]
        phys_u   = [f.getPhysicalDimension(i)     for i in range(n)]
        phys_min = [f.getPhysicalMinimum(i)       for i in range(n)]
        phys_max = [f.getPhysicalMaximum(i)       for i in range(n)]
        dig_min  = [f.getDigitalMinimum(i)        for i in range(n)]
        dig_max  = [f.getDigitalMaximum(i)        for i in range(n)]
        pre      = [f.getPrefilter(i)             for i in range(n)]
        trans    = [f.getTransducer(i)            for i in range(n)]
        ns       = f.getNSamples()  # returns a list for all signals

        df = pd.DataFrame({
            "Channel #":            np.arange(1, n+1, dtype=np.int32),
            "Label":                labels,
            "Sample Frequency (Hz)": sfs,
            "Physical Unit":        phys_u,
            "Physical Min":         phys_min,
            "Physical Max":         phys_max,
            "Digital Min":          dig_min,
            "Digital Max":          dig_max,
            "Prefilter":            pre,
            "Transducer":           trans,
            "Num Samples":          ns,
            "Signal Length (s)":    np.array(ns, dtype=np.float64) / np.array(sfs, dtype=np.float64),
        })
        return df
    except Exception as e:
        raise RuntimeError(f"Error reading EDF file: {e}")
    finally:
        try:
            if f is not None:
                f.close()
        except Exception:
            pass


def read_edf_to_dataframes(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Fast reader: returns a dict of DataFrames (one per channel).
    Columns per DF:
      - 'Relative Time (s)'  (vectorized arange: 1/Fs, 2/Fs, ...)
      - 'Absolute Time'      (vectorized numpy datetime64[ns])
      - '<Unique Label>'     (signal, float32)
    """
    def make_unique_labels_only(f: pyedflib.EdfReader):
        """Create unique channel labels with minimal overhead (no mapping DF)."""
        sig_headers = f.getSignalHeaders()
        buckets = defaultdict(list)
        for i, h in enumerate(sig_headers):
            label = (h.get('label') or '').strip() or f'chan{i+1}'
            transducer = (h.get('transducer') or '').strip()
            buckets[label].append((i, label, transducer, h))

        unique_labels = [None] * len(sig_headers)
        for base_label, items in buckets.items():
            if len(items) == 1:
                i, _, transducer, _ = items[0]
                unique_labels[i] = f"{base_label} ({transducer})" if transducer else base_label
            else:
                seen_trans = set()
                all_have_trans = all((t or '').strip() for (_, _, t, _) in items)
                for i, _, transducer, _ in items:
                    if all_have_trans and transducer not in seen_trans:
                        unique_labels[i] = f"{base_label} ({transducer})"
                        seen_trans.add(transducer)
                    else:
                        unique_labels[i] = f"{base_label} ({transducer}) [ch{i+1}]" if transducer else f"{base_label} [ch{i+1}]"
        return unique_labels

    f = None
    try:
        f = pyedflib.EdfReader(file_path)
        start_dt = f.getStartdatetime()  # naive datetime from pyedflib
        # Normalize to UTC-aware for consistency (optional)
        if start_dt.tzinfo is None:
            # Treat as UTC; change here if your files are local time
            start_dt = start_dt.replace(tzinfo=timezone.utc)

        start_ns = np.datetime64(start_dt)  # numpy datetime64[ns]

        n_signals = f.signals_in_file
        unique_labels = make_unique_labels_only(f)

        # Pre-extract reusable header fields to reduce repeated dict lookups
        sig_headers = f.getSignalHeaders()

        channel_dataframes: Dict[str, pd.DataFrame] = {}

        for i in range(n_signals):
            h = sig_headers[i]
            fs = h.get('sample_frequency')
            if not fs or fs <= 0:
                continue  # skip invalid sampling rates

            label = unique_labels[i]

            # readSignal returns a NumPy array already; downcast to float32 to speed DF creation and save memory
            sig = f.readSignal(i).astype(np.float32, copy=False)
            n = sig.size
            if n == 0:
                continue

            # Vectorized time bases
            # Relative time (s): 1/Fs, 2/Fs, ..., n/Fs (float32 is fine here)
            rel_t = (np.arange(1, n+1, dtype=np.float64) / float(fs)).astype(np.float32)

            # Absolute time: start_ns + rel_t in ns (build once as int64 ns)
            # Convert seconds to ns as int64 to preserve exact increments
            rel_ns = (rel_t.astype(np.float64) * 1e9).astype('int64')
            abs_t = (start_ns + rel_ns.astype('timedelta64[ns]'))  # numpy datetime64[ns]
            # Pandas accepts datetime64[ns] directly without Python datetime loops
            abs_t = pd.to_datetime(abs_t)  # keeps ns precision

            # Build DataFrame from arrays (no Python loops)
            df = pd.DataFrame({
                'Relative Time (s)': rel_t,
                'Absolute Time': abs_t,
                label: sig
            })

            # Preserve useful header info in attrs (nice for QC later)
            df.attrs['physical_min'] = h.get('physical_min')
            df.attrs['physical_max'] = h.get('physical_max')
            df.attrs['digital_min']  = h.get('digital_min')
            df.attrs['digital_max']  = h.get('digital_max')
            df.attrs['sample_frequency'] = fs

            channel_dataframes[label] = df

        return channel_dataframes

    except Exception as e:
        raise RuntimeError(f"Error reading EDF file '{file_path}': {e}")

    finally:
        try:
            if f is not None:
                f.close()
        except Exception:
            pass
