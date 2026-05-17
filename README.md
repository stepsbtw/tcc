# IPqM-Fall Dataset

DOI: [10.5281/zenodo.12760390](https://doi.org/10.5281/zenodo.12760390)

## 1. Download

Download `IPqM-Fall.zip` from the Zenodo record above and extract it in the project root so the folder `IPqM-Fall/`

## 2. Preprocessing

```bash
python scripts/fix_dataset.py --root IPqM-Fall
```
(drop stale columns, recompute `trial`, rename `Magnitude`, sort sampling, and check ID5 timestamp offsets).

```bash
python scripts/preprocess.py
```
(`csv2parquet`, `flatten`, `segment`, `downsample`, `windows`, `labelstudio`, `app`).

### Manually

- `python scripts/fix_dataset.py --root IPqM-Fall` — dataset fixes applied to the raw files.
- `python scripts/csv_to_parquet.py` — convert CSVs to parquet (writes to `IPqM-Fall/OLD/parquet/`).
- `python scripts/flatten.py` — assemble trial segments into `IPqM-Fall/trials/` (uses sampling labels).
- `python scripts/segment_trials.py` — alternative segmentation if needed.
- `python downsample_trials.py` — resample trials to 90 Hz (writes `IPqM-Fall/trials_90hz/`).
- `python generate_windows.py` — create `window_labels.csv` (windows metadata used by the review app).
- `python scripts/window_trials.py` — export windows as a Label Studio JSON.

## 3. Review Labels

```bash
streamlit run app.py
```