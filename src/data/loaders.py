"""
PyTorch DataLoader for glaucoma progression dataset.
Supports GRAPE, UWHVF, and custom multi-modal datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import polars as pl
from typing import Any


class GlaucomaDataset(Dataset):
    """
    Multi-modal glaucoma dataset with longitudinal visual field data.

    Expected data structure (from preprocessing pipeline):
    - data/processed/patients.parquet  — patient metadata & progression labels
    - data/processed/vf_series.parquet — longitudinal VF sensitivity values
    - data/processed/images/           — fundus/OCT images per patient
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",                  # train | val | test
        n_vf_points: int = 54,                 # 24-2 VF has 54 test points
        max_visits: int = 10,                  # max VF visits to use
        n_clinical_features: int = 5,          # available in UWHVF: age, gender, n_visits, follow_up, md_baseline
        image_size: tuple[int, int] = (224, 224),
        transform: Any = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.n_vf_points = n_vf_points
        self.max_visits = max_visits
        self.n_clinical = n_clinical_features
        self.image_size = image_size
        self.transform = transform

        self._load_metadata()

    def _load_metadata(self) -> None:
        split_file = self.data_dir / "splits" / f"{self.split}_ids.txt"
        # Also check legacy location (data/processed/)
        if not split_file.exists():
            split_file = self.data_dir / "processed" / f"{self.split}_ids.txt"
        patients_file = self.data_dir / "processed" / "patients.parquet"

        if not patients_file.exists():
            raise FileNotFoundError(
                f"Processed data not found at {patients_file}. "
                "Run: python scripts/preprocess.py first."
            )

        all_patients = pl.read_parquet(patients_file)

        if split_file.exists():
            patient_ids = split_file.read_text().strip().splitlines()
            self.patients = all_patients.filter(pl.col("patient_id").is_in(patient_ids))
        else:
            self.patients = all_patients

        self.patient_ids = self.patients["patient_id"].to_list()

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pid = self.patient_ids[idx]
        row = self.patients.filter(pl.col("patient_id") == pid).row(0, named=True)

        # 1. Load image (fundus or OCT)
        image = self._load_image(pid)

        # 2. Load VF time series
        vf_series, vf_length = self._load_vf_series(pid)

        # 3. Clinical features — pre-normalised [0,1] columns from preprocessing
        clinical = torch.tensor(
            [
                row.get("age_baseline_norm",  row.get("age_baseline", 0.0)),
                row.get("gender_norm",         row.get("gender", 0.0)),
                row.get("n_visits_norm",       row.get("n_visits", 0.0)),
                row.get("follow_up_norm",      row.get("follow_up_years", 0.0)),
                row.get("md_baseline_norm",    row.get("md_baseline", 0.0)),
            ],
            dtype=torch.float32,
        )

        # 4. Progression labels (what we predict)
        labels = {
            "md_rate": torch.tensor(row.get("md_rate_db_year", 0.0), dtype=torch.float32),
            "vfi_rate": torch.tensor(row.get("vfi_rate_pct_year", 0.0), dtype=torch.float32),
        }

        return {
            "image": image,
            "vf_series": vf_series,
            "vf_length": torch.tensor(vf_length, dtype=torch.long),
            "clinical": clinical,
            **labels,
        }

    def _load_image(self, patient_id: str) -> torch.Tensor:
        img_path = self.data_dir / "processed" / "images" / f"{patient_id}.pt"
        if img_path.exists():
            return torch.load(img_path, weights_only=True)
        # Return zeros if image not available
        return torch.zeros(3, *self.image_size)

    def _load_vf_series(self, patient_id: str) -> tuple[torch.Tensor, int]:
        vf_path = self.data_dir / "processed" / f"vf_{patient_id}.npy"
        if vf_path.exists():
            vf = np.load(vf_path)  # (T, n_points)
            n_visits = min(len(vf), self.max_visits)
            padded = np.zeros((self.max_visits, self.n_vf_points), dtype=np.float32)
            padded[:n_visits] = vf[:n_visits]
            return torch.tensor(padded), n_visits
        return torch.zeros(self.max_visits, self.n_vf_points), 0


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs: Any,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders."""
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = GlaucomaDataset(data_dir=data_dir, split=split, **dataset_kwargs)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders
