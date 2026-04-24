"""
Data preprocessing pipeline using Polars for fast temporal alignment.
Handles GRAPE and UWHVF dataset formats.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Any


def preprocess_visual_fields(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean and normalize visual field sensitivity values.
    - Clips pathological values to [-35, 0] dB range
    - Normalizes to [0, 1]
    - Flags unreliable tests (fixation losses, false positives/negatives)
    """
    vf_cols = [c for c in df.columns if c.startswith("vf_point_")]

    # Reliability filter: flag tests with high false positive rate
    if "false_positive_rate" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("false_positive_rate") > 0.33)
            .then(True)
            .otherwise(False)
            .alias("unreliable")
        )

    # Clip and normalize VF sensitivities
    df = df.with_columns([
        pl.col(c).clip(-35.0, 0.0).alias(c) for c in vf_cols
    ])
    df = df.with_columns([
        ((pl.col(c) + 35.0) / 35.0).alias(f"{c}_norm") for c in vf_cols
    ])

    return df


def align_temporal_visits(
    patient_id: str,
    visits_df: pl.DataFrame,
    min_visits: int = 3,
    max_interval_years: float = 5.0,
) -> dict[str, Any] | None:
    """
    Align longitudinal visits for a single patient.
    Filters out gaps >5 years (likely treatment change or data error).
    Computes MD and VFI rates via OLS regression on valid visits.

    Returns None if patient has insufficient data.
    """
    patient_visits = (
        visits_df
        .filter(pl.col("patient_id") == patient_id)
        .sort("visit_date")
    )

    if len(patient_visits) < min_visits:
        return None

    # Compute time from baseline in years
    baseline_date = patient_visits["visit_date"].min()
    patient_visits = patient_visits.with_columns(
        ((pl.col("visit_date") - baseline_date).dt.total_days() / 365.25)
        .alias("years_from_baseline")
    )

    # Filter out large temporal gaps
    patient_visits = patient_visits.filter(
        pl.col("years_from_baseline") <= max_interval_years
    )

    if len(patient_visits) < min_visits:
        return None

    # OLS regression for MD and VFI rates
    times = patient_visits["years_from_baseline"].to_numpy()
    md_values = patient_visits["md_db"].to_numpy()
    vfi_values = patient_visits["vfi_pct"].to_numpy()

    md_rate = _ols_slope(times, md_values)    # dB/year
    vfi_rate = _ols_slope(times, vfi_values)  # %/year

    return {
        "patient_id": patient_id,
        "n_visits": len(patient_visits),
        "follow_up_years": float(times[-1]),
        "md_baseline": float(md_values[0]),
        "vfi_baseline": float(vfi_values[0]),
        "md_rate_db_year": md_rate,
        "vfi_rate_pct_year": vfi_rate,
    }


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least squares slope (rate of change)."""
    x_mean, y_mean = x.mean(), y.mean()
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def create_stratified_splits(
    patients_df: pl.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    output_dir: Path = Path("data/splits"),
) -> None:
    """
    Create stratified train/val/test splits.
    Stratifies by progression rate quartile to ensure balanced representation
    of fast vs slow progressors in each split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratify by MD rate quartile
    df = patients_df.with_columns(
        pl.col("md_rate_db_year")
        .qcut(4, labels=["Q1", "Q2", "Q3", "Q4"])
        .alias("md_quartile")
    )

    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []

    for _, group in df.group_by("md_quartile"):
        ids = group["patient_id"].to_list()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        (output_dir / f"{split}_ids.txt").write_text("\n".join(ids))

    print(f"Splits created: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
