import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import re

TRIALS_ROOT = Path("IPqM-Fall/trials_90hz")

WINDOW_SEC = 2
STRIDE_SEC = 1
FS = 90

WINDOW_SIZE = WINDOW_SEC * FS
STRIDE = STRIDE_SEC * FS

rows = []

trial_files = sorted(TRIALS_ROOT.glob("*.parquet"))

start_time = time.time()

for parquet_file in tqdm(
    trial_files,
    desc="Processing trials",
    unit="file"
):

    df = pd.read_parquet(parquet_file)

    if len(df) < WINDOW_SIZE:
        continue

    stem = parquet_file.stem

    # Extract activity label from filename
    # Examples:
    # ADL_11
    # OM_3
    # FALL_2

    match = re.search(
        r"(ADL_\d+|OM_\d+|FALL_\d+)",
        stem
    )

    if match:
        initial_label = match.group(1)
    else:
        initial_label = "UNKNOWN"

    n_windows = (
        (len(df) - WINDOW_SIZE) // STRIDE
    ) + 1

    for i in range(n_windows):

        start = i * STRIDE
        end = start + WINDOW_SIZE

        rows.append({
            "file": parquet_file.name,
            "window_id": f"{stem}_win_{start}",
            "start_idx": start,
            "end_idx": end,
            "label": initial_label,
            "reviewed": False
        })

elapsed = time.time() - start_time

meta = pd.DataFrame(rows)

meta.to_csv("window_labels.csv", index=False)

print("\n========================")
print(f"Trials processed : {len(trial_files)}")
print(f"Windows created  : {len(meta)}")
print(f"Elapsed time     : {elapsed:.2f} sec")
print("========================")

print(meta.head())