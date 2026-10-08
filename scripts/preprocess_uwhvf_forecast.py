"""
Forecast preprocessing for UWHVF.

Fixes the Scientific Reports critique: labels are the OLS slope of
SUBSEQUENT visits; the model may see only the first K visits.

Input:  data/raw/uwhvf/alldata.json
Output: data/forecast/processed/patients.parquet
        data/forecast/processed/vf_{pid}.npy   (K, 54) — observation window only
        data/forecast/splits/{train,val,test}_ids.txt
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_JSON = ROOT / "data" / "raw" / "uwhvf" / "alldata.json"
PROC_DIR = ROOT / "data" / "forecast" / "processed"
SPLITS_DIR = ROOT / "data" / "forecast" / "splits"

PROC_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

K_INPUT = 3
MIN_FUTURE_VISITS = 3
MIN_FUTURE_YEARS = 1.0
RATE_LO, RATE_HI = -10.0, 5.0
N_VF_POINTS = 54
SENTINEL = 90.0


def md_from_td(td_seq: list[float]) -> float:
    td = np.asarray(td_seq, dtype=np.float64)
    valid = td[td < SENTINEL]
    return float(valid.mean()) if len(valid) > 0 else np.nan


def normalize_td(td_seq: list[float]) -> np.ndarray:
    td = np.asarray(td_seq, dtype=np.float32)
    td = np.where(td >= SENTINEL, 0.0, td)
    td = np.clip(td, -35.0, 0.0)
    return (td + 35.0) / 35.0


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm) ** 2).sum()
    return float(num / den) if den > 1e-10 else 0.0


def main() -> None:
    print(f"Loading {RAW_JSON} …")
    with open(RAW_JSON) as f:
        raw = json.load(f)
    patients_raw = raw["data"]
    print(f"  Patients in raw JSON: {len(patients_raw):,}")

    pe_visits: dict[str, list[dict]] = defaultdict(list)
    for pid_str, patient in patients_raw.items():
        gender = 1 if str(patient.get("gender", "F")).upper() == "M" else 0
        for eye in ("L", "R"):
            visits = patient.get(eye)
            if not isinstance(visits, list):
                continue
            for vi, visit in enumerate(visits):
                age = visit.get("age")
                td = visit.get("td_seq") or visit.get("td")
                if age is None or td is None or len(td) != N_VF_POINTS:
                    continue
                pe_id = f"{pid_str}_{eye}"
                pe_visits[pe_id].append(
                    {"age": float(age), "gender": gender, "td_seq": td, "visit_idx": vi}
                )
    print(f"  Patient-eyes with ≥1 visit: {len(pe_visits):,}")

    records = []
    vf_series = {}
    n_too_short = n_future_short = n_rate_out = n_nan = 0

    for pe_id, visits in pe_visits.items():
        if len(visits) < K_INPUT + MIN_FUTURE_VISITS:
            n_too_short += 1
            continue
        visits_sorted = sorted(visits, key=lambda v: v["age"])
        ages = np.array([v["age"] for v in visits_sorted], dtype=np.float64)
        mds = np.array([md_from_td(v["td_seq"]) for v in visits_sorted])
        if np.any(np.isnan(mds)):
            n_nan += 1
            continue

        obs = visits_sorted[:K_INPUT]
        fut = visits_sorted[K_INPUT:]
        ages_obs = ages[:K_INPUT]
        ages_fut = ages[K_INPUT:]
        mds_obs = mds[:K_INPUT]
        mds_fut = mds[K_INPUT:]

        future_span = float(ages_fut[-1] - ages_fut[0])
        if len(fut) < MIN_FUTURE_VISITS or future_span < MIN_FUTURE_YEARS:
            n_future_short += 1
            continue

        times_fut = ages_fut - ages_fut[0]
        vfi_fut = np.clip(100.0 + mds_fut * 2.0, 0.0, 100.0)
        md_rate = ols_slope(times_fut, mds_fut)
        if not (RATE_LO <= md_rate <= RATE_HI):
            n_rate_out += 1
            continue
        vfi_rate = ols_slope(times_fut, vfi_fut)

        times_obs = ages_obs - ages_obs[0]
        vfi_obs = np.clip(100.0 + mds_obs * 2.0, 0.0, 100.0)
        md_rate_early = ols_slope(times_obs, mds_obs)
        vfi_rate_early = ols_slope(times_obs, vfi_obs)

        input_span = float(ages_obs[-1] - ages_obs[0])
        age_norm = float(np.clip((ages_obs[0] - 40.0) / 50.0, 0.0, 1.0))
        gender = int(obs[0]["gender"])
        nv_norm = float(np.clip((K_INPUT - 3) / 17.0, 0.0, 1.0))
        fu_norm = float(np.clip(input_span / 20.0, 0.0, 1.0))
        mdb_norm = float(np.clip((mds_obs[0] + 35.0) / 35.0, 0.0, 1.0))

        records.append(
            {
                "patient_id": pe_id,
                "age_baseline": float(ages_obs[0]),
                "gender": gender,
                "n_visits": K_INPUT,
                "n_total_visits": len(visits_sorted),
                "n_future_visits": len(fut),
                "input_span_years": input_span,
                "future_span_years": future_span,
                "follow_up_years": input_span,
                "md_baseline": float(mds_obs[0]),
                "vfi_baseline": float(vfi_obs[0]),
                "md_rate_db_year": md_rate,
                "vfi_rate_pct_year": vfi_rate,
                "md_rate_early_ols": md_rate_early,
                "vfi_rate_early_ols": vfi_rate_early,
                "age_baseline_norm": age_norm,
                "gender_norm": float(gender),
                "n_visits_norm": nv_norm,
                "follow_up_norm": fu_norm,
                "md_baseline_norm": mdb_norm,
            }
        )
        vf_series[pe_id] = np.stack(
            [normalize_td(v["td_seq"]) for v in obs], axis=0
        ).astype(np.float32)

    print("\nFilter counts:")
    print(f"  too few total visits: {n_too_short:,}")
    print(f"  future window too short: {n_future_short:,}")
    print(f"  NaN MD: {n_nan:,}")
    print(f"  future rate outlier: {n_rate_out:,}")
    print(f"  retained: {len(records):,}")

    patients_df = pl.DataFrame(records)
    patients_df.write_parquet(PROC_DIR / "patients.parquet")
    print(f"\nSaved {PROC_DIR / 'patients.parquet'}")

    md_rates = patients_df["md_rate_db_year"].to_numpy()
    print(
        f"  Future MD rate — mean={md_rates.mean():.3f}  std={md_rates.std():.3f}  "
        f"range=[{md_rates.min():.2f}, {md_rates.max():.2f}]"
    )
    print(
        f"  Fast progressors (future MD < -1 dB/yr): "
        f"{(md_rates < -1).sum()} / {len(md_rates)} ({(md_rates < -1).mean()*100:.1f}%)"
    )

    for pe_id, arr in vf_series.items():
        np.save(PROC_DIR / f"vf_{pe_id}.npy", arr)
    print(f"Saved {len(vf_series):,} observation-window VF files")

    # Patient-level split so fellow eyes cannot leak across partitions.
    df = patients_df.with_columns(
        pl.col("patient_id").str.replace(r"_[LR]$", "").alias("patient")
    )
    patient_tbl = df.group_by("patient").agg(
        pl.col("md_rate_db_year").mean().alias("md_mean")
    ).with_columns(
        pl.col("md_mean").qcut(4, labels=["Q1", "Q2", "Q3", "Q4"]).alias("md_quartile")
    )
    rng = np.random.default_rng(42)
    train_p, val_p, test_p = set(), set(), set()
    for (_,), group in patient_tbl.group_by("md_quartile"):
        ids = group["patient"].to_list()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_p.update(ids[:n_train])
        val_p.update(ids[n_train : n_train + n_val])
        test_p.update(ids[n_train + n_val :])

    def eyes_for(patients: set[str]) -> list[str]:
        return df.filter(pl.col("patient").is_in(list(patients)))["patient_id"].to_list()

    train_ids, val_ids, test_ids = eyes_for(train_p), eyes_for(val_p), eyes_for(test_p)

    (SPLITS_DIR / "train_ids.txt").write_text("\n".join(train_ids))
    (SPLITS_DIR / "val_ids.txt").write_text("\n".join(val_ids))
    (SPLITS_DIR / "test_ids.txt").write_text("\n".join(test_ids))
    print(f"\nSplits: train={len(train_ids):,}  val={len(val_ids):,}  test={len(test_ids):,}")
    print(
        f"Patients: train={len(train_p):,}  val={len(val_p):,}  test={len(test_p):,}  "
        f"unique={df['patient'].n_unique():,}"
    )

    REPORTS = ROOT / "reports" / "forecast"
    REPORTS.mkdir(parents=True, exist_ok=True)
    rates = patients_df["md_rate_db_year"].to_numpy()
    cohort = {
        "n_eyes": len(patients_df),
        "n_patients": int(df["patient"].n_unique()),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_train_patients": len(train_p),
        "n_val_patients": len(val_p),
        "n_test_patients": len(test_p),
        "k_input": K_INPUT,
        "min_future_visits": MIN_FUTURE_VISITS,
        "min_future_years": MIN_FUTURE_YEARS,
        "md_rate_mean": float(rates.mean()),
        "md_rate_std": float(rates.std()),
        "n_fast": int((rates < -1).sum()),
        "pct_fast": float((rates < -1).mean() * 100),
        "n_total_visits_mean": float(patients_df["n_total_visits"].mean()),
        "n_total_visits_std": float(patients_df["n_total_visits"].std()),
        "n_future_visits_mean": float(patients_df["n_future_visits"].mean()),
        "input_span_median": float(patients_df["input_span_years"].median()),
        "future_span_median": float(patients_df["future_span_years"].median()),
        "future_span_iqr": [
            float(patients_df["future_span_years"].quantile(0.25)),
            float(patients_df["future_span_years"].quantile(0.75)),
        ],
        "md_baseline_mean": float(patients_df["md_baseline"].mean()),
        "md_baseline_std": float(patients_df["md_baseline"].std()),
    }
    (REPORTS / "cohort.json").write_text(json.dumps(cohort, indent=2))
    print("Forecast preprocessing complete.")


if __name__ == "__main__":
    main()
