import pyedflib
import pandas as pd
from datetime import timedelta
from collections import defaultdict
from typing import Dict

def read_edf_metadata(file_path):
    """
    Read EDF file metadata and return it as a pandas DataFrame.
    No printing — only returns the DataFrame.
    """
    try:
        f = pyedflib.EdfReader(file_path)
        n_channels = f.signals_in_file

        # Collect metadata per channel
        metadata = []
        for i in range(n_channels):
            metadata.append({
                "Channel #": i + 1,
                "Label": f.getLabel(i),
                "Sample Frequency (Hz)": f.getSampleFrequency(i),
                "Physical Unit": f.getPhysicalDimension(i),
                "Physical Min": f.getPhysicalMinimum(i),
                "Physical Max": f.getPhysicalMaximum(i),
                "Digital Min": f.getDigitalMinimum(i),
                "Digital Max": f.getDigitalMaximum(i),
                "Prefilter": f.getPrefilter(i),
                "Transducer": f.getTransducer(i),
                "Num Samples": f.getNSamples()[i],
                "Signal Length (s)": f.getNSamples()[i] / f.getSampleFrequency(i),
            })

        df = pd.DataFrame(metadata)
        return df

    except Exception as e:
        raise RuntimeError(f"Error reading EDF file: {e}")
    finally:
        f.close()



def read_edf_to_dataframes(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Read an EDF file and return a dict of pandas DataFrames, one per channel,
    using unique, disambiguated labels as column names.

    Each DataFrame contains:
      - 'Relative Time (s)'  (starts at 1/Fs, 2/Fs, ..., like your original code)
      - 'Absolute Time'
      - '<Unique Label>'     (the signal samples)

    Parameters
    ----------
    file_path : str
        Path to the EDF file.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from unique channel label -> DataFrame.
    """
    def make_unique_labels(f):
        """Create unique channel labels with transducer suffixes when helpful."""
        sig_headers = f.getSignalHeaders()
        buckets = defaultdict(list)

        for i, h in enumerate(sig_headers):
            label = (h.get('label') or '').strip() or f'chan{i+1}'
            transducer = (h.get('transducer') or '').strip()
            buckets[label].append((i, label, transducer, h))

        unique_labels = [None] * len(sig_headers)
        map_rows = []

        for base_label, items in buckets.items():
            if len(items) == 1:
                i, _, transducer, h = items[0]
                nice = f"{base_label} ({transducer})" if transducer else base_label
                unique_labels[i] = nice
                map_rows.append({
                    "idx": i+1,
                    "label": base_label,
                    "unique_label": nice,
                    "transducer": transducer,
                    "sf": h.get('sample_frequency'),
                    "phys_min": h.get('physical_min'),
                    "phys_max": h.get('physical_max'),
                    "dig_min": h.get('digital_min'),
                    "dig_max": h.get('digital_max'),
                    "n_samples": h.get('n_samples')
                })
            else:
                seen_trans = set()
                all_have_trans = all((t or '').strip() for (_, _, t, _) in items)
                for k, (i, _, transducer, h) in enumerate(items, start=1):
                    if all_have_trans and transducer not in seen_trans:
                        nice = f"{base_label} ({transducer})"
                        seen_trans.add(transducer)
                    else:
                        nice = f"{base_label} [ch{i+1}]" if not transducer else f"{base_label} ({transducer}) [ch{i+1}]"
                    unique_labels[i] = nice
                    map_rows.append({
                        "idx": i+1,
                        "label": base_label,
                        "unique_label": nice,
                        "transducer": transducer,
                        "sf": h.get('sample_frequency'),
                        "phys_min": h.get('physical_min'),
                        "phys_max": h.get('physical_max'),
                        "dig_min": h.get('digital_min'),
                        "dig_max": h.get('digital_max'),
                        "n_samples": h.get('n_samples')
                    })

        return unique_labels, pd.DataFrame(map_rows).sort_values("idx")

    f = None
    channel_dataframes = {}
    try:
        f = pyedflib.EdfReader(file_path)
        start_datetime = f.getStartdatetime()
        n_signals = f.signals_in_file

        unique_labels, _ = make_unique_labels(f)

        for i in range(n_signals):
            header = f.getSignalHeader(i)
            sampling_frequency = header.get('sample_frequency')
            if sampling_frequency is None or sampling_frequency <= 0:
                # Skip channels with invalid sampling rate
                continue

            unique_label = unique_labels[i]
            channel_data = f.readSignal(i)

            n_samples = len(channel_data)
            interval = 1.0 / sampling_frequency

            # Relative time starting at 1/Fs (matches your original approach)
            relative_time = [(j + 1) * interval for j in range(n_samples)]
            absolute_time = [start_datetime + timedelta(seconds=rt) for rt in relative_time]

            df = pd.DataFrame({
                'Relative Time (s)': relative_time,
                'Absolute Time': absolute_time,
                unique_label: channel_data
            })
            channel_dataframes[unique_label] = df

        return channel_dataframes

    except Exception as e:
        raise RuntimeError(f"Error reading EDF file '{file_path}': {e}")

    finally:
        try:
            if f is not None:
                f.close()
        except Exception:
            pass
