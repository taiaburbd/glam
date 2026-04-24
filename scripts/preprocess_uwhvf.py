"""
Preprocess real UWHVF dataset for GLAM model training.

Input:  data/raw/uwhvf/alldata.json
Output: data/processed/patients.parquet
        data/processed/vf_{pid}.npy   (T, 54) float32 — normalized TD values
        data/splits/train_ids.txt
        data/splits/val_ids.txt
        data/splits/test_ids.txt

Filters applied:
  - ≥ 3 visits per patient-eye
  - ≥ 1.0 year follow-up (age[-1] - age[0])
  - MD rate within [-10, 5] dB/yr  (removes <1% outliers)

Clinical features (5):
  age_baseline_norm     (age - 40) / 50
  gender                0=F, 1=M
  n_visits_norm         (n - 3) / 17
  follow_up_norm        follow_up_years / 20
  md_baseline_norm      (md_baseline + 35) / 35

VF features: td_seq clipped to [-35, 0] then scaled to [0, 1]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_JSON   = ROOT / "data" / "raw" / "uwhvf" / "alldata.json"
PROC_DIR   = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

PROC_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

MIN_VISITS    = 3
MIN_FOLLOW_UP = 1.0     # years
RATE_LO, RATE_HI = -10.0, 5.0
N_VF_POINTS   = 54
SENTINEL      = 90.0    # td values >= SENTINEL are "untested" placeholders


# ── helpers ───────────────────────────────────────────────────────────────────

def md_from_td(td_seq: list[float]) -> float:
    td = np.asarray(td_seq, dtype=np.float64)
    valid = td[td < SENTINEL]
    return float(valid.mean()) if len(valid) > 0 else np.nan


def normalize_td(td_seq: list[float]) -> np.ndarray:
    """Clip TD to [-35, 0] dB, scale to [0, 1]. Sentinels → 0.0."""
    td = np.asarray(td_seq, dtype=np.float32)
    td = np.where(td >= SENTINEL, 0.0, td)      # sentinel → 0
    td = np.clip(td, -35.0, 0.0)
    td = (td + 35.0) / 35.0                     # [0, 1]
    return td


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm) ** 2).sum()
    return float(num / den) if den > 1e-10 else 0.0


# ── load raw data ─────────────────────────────────────────────────────────────

print(f"Loading {RAW_JSON} …")
with open(RAW_JSON) as f:
    raw = json.load(f)

patients_raw = raw["data"]
print(f"  Patients in raw JSON: {len(patients_raw):,}")

# ── flatten to (patient_id, eye, visit) records ───────────────────────────────

pe_visits: dict[str, list[dict]] = defaultdict(list)

for pid_str, patient in patients_raw.items():
    gender = 1 if str(patient.get("gender", "F")).upper() == "M" else 0
    for eye in ("L", "R"):
        visits = patient.get(eye)
        if not isinstance(visits, list):
            continue
        for vi, visit in enumerate(visits):
            age  = visit.get("age")
            td   = visit.get("td_seq")
            if age is None or td is None or len(td) != N_VF_POINTS:
                continue
            pe_id = f"{pid_str}_{eye}"
            pe_visits[pe_id].append({"age": float(age), "gender": gender,
                                     "td_seq": td, "visit_idx": vi})

print(f"  Patient-eyes with ≥1 visit: {len(pe_visits):,}")

# ── compute progression labels ────────────────────────────────────────────────

records   = []
vf_series = {}

for pe_id, visits in pe_visits.items():
    if len(visits) < MIN_VISITS:
        continue

    visits_sorted = sorted(visits, key=lambda v: v["age"])
    ages  = np.array([v["age"] for v in visits_sorted], dtype=np.float64)
    times = ages - ages[0]                         # years from baseline

    follow_up = float(times[-1])
    if follow_up < MIN_FOLLOW_UP:
        continue

    mds = np.array([md_from_td(v["td_seq"]) for v in visits_sorted])
    if np.any(np.isnan(mds)):
        continue

    md_rate  = ols_slope(times, mds)
    if not (RATE_LO <= md_rate <= RATE_HI):
        continue

    # VFI proxy: linear approximation from MD proxy
    vfi_vals = np.clip(100.0 + mds * 2.0, 0.0, 100.0)
    vfi_rate = ols_slope(times, vfi_vals)

    # Normalised clinical features
    age_norm  = float(np.clip((ages[0] - 40.0) / 50.0, 0.0, 1.0))
    gender    = int(visits_sorted[0]["gender"])
    nv_norm   = float(np.clip((len(visits_sorted) - 3) / 17.0, 0.0, 1.0))
    fu_norm   = float(np.clip(follow_up / 20.0, 0.0, 1.0))
    mdb_norm  = float(np.clip((mds[0] + 35.0) / 35.0, 0.0, 1.0))

    records.append({
        "patient_id":           pe_id,
        # raw for reporting
        "age_baseline":         float(ages[0]),
        "gender":               gender,
        "n_visits":             len(visits_sorted),
        "follow_up_years":      follow_up,
        "md_baseline":          float(mds[0]),
        "vfi_baseline":         float(vfi_vals[0]),
        "md_rate_db_year":      md_rate,
        "vfi_rate_pct_year":    vfi_rate,
        # normalised clinical (used directly by model)
        "age_baseline_norm":    age_norm,
        "gender_norm":          float(gender),
        "n_visits_norm":        nv_norm,
        "follow_up_norm":       fu_norm,
        "md_baseline_norm":     mdb_norm,
    })

    # VF series: (T, 54) float32
    vf_arr = np.stack(
        [normalize_td(v["td_seq"]) for v in visits_sorted], axis=0
    ).astype(np.float32)
    vf_series[pe_id] = vf_arr

print(f"\nAfter filters ({MIN_VISITS}+ visits, {MIN_FOLLOW_UP}+ yr, rate ∈ [{RATE_LO},{RATE_HI}]):")
print(f"  Valid patient-eyes: {len(records):,}")

# ── save patients.parquet ─────────────────────────────────────────────────────

patients_df = pl.DataFrame(records)
patients_df.write_parquet(PROC_DIR / "patients.parquet")
print(f"\nSaved patients.parquet  ({len(patients_df):,} rows)")

# descriptive stats
md_rates = patients_df["md_rate_db_year"].to_numpy()
print(f"  MD rate  — mean={md_rates.mean():.3f}  std={md_rates.std():.3f}  "
      f"range=[{md_rates.min():.2f}, {md_rates.max():.2f}]")
print(f"  Fast progressors (MD < -1 dB/yr): "
      f"{(md_rates < -1).sum()} / {len(md_rates)} "
      f"({(md_rates < -1).mean()*100:.1f}%)")
n_visits_arr = patients_df["n_visits"].to_numpy()
print(f"  Visits/eye — mean={n_visits_arr.mean():.1f}  "
      f"median={np.median(n_visits_arr):.0f}  max={n_visits_arr.max()}")
fu_arr = patients_df["follow_up_years"].to_numpy()
print(f"  Follow-up  — mean={fu_arr.mean():.1f}yr  median={np.median(fu_arr):.1f}yr")

# ── save VF series ────────────────────────────────────────────────────────────

for pe_id, arr in vf_series.items():
    np.save(PROC_DIR / f"vf_{pe_id}.npy", arr)

print(f"\nSaved {len(vf_series):,} VF series files to {PROC_DIR}/")

# ── stratified train / val / test splits ─────────────────────────────────────

df = patients_df.with_columns(
    pl.col("md_rate_db_year")
      .qcut(4, labels=["Q1", "Q2", "Q3", "Q4"])
      .alias("md_quartile")
)

rng = np.random.default_rng(42)
train_ids, val_ids, test_ids = [], [], []

for (q,), group in df.group_by("md_quartile"):
    ids = group["patient_id"].to_list()
    rng.shuffle(ids)
    n       = len(ids)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_ids.extend(ids[:n_train])
    val_ids  .extend(ids[n_train : n_train + n_val])
    test_ids .extend(ids[n_train + n_val :])

(SPLITS_DIR / "train_ids.txt").write_text("\n".join(train_ids))
(SPLITS_DIR / "val_ids.txt")  .write_text("\n".join(val_ids))
(SPLITS_DIR / "test_ids.txt") .write_text("\n".join(test_ids))

print(f"\nSplits (stratified by MD-rate quartile):")
print(f"  train : {len(train_ids):,}")
print(f"  val   : {len(val_ids):,}")
print(f"  test  : {len(test_ids):,}")
print(f"\nPreprocessing complete.")
