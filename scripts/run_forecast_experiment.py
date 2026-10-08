"""
Train and evaluate GLAM on the forecast task:
  input  = first 3 VF visits
  target = OLS MD/VFI slope of subsequent visits only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import GlaucomaDataset
from src.losses.progression import NegativeLogLikelihoodLoss, WeightedDualRegressionLoss
from src.models.backbones import VisualFieldEncoder
from src.models.fusion import AttentionFusion
from src.models.heads import GLAMModel, ProgressionHead
from src.models.temporal import BidirectionalLSTM
from src.utils.metrics import compute_all_metrics
from src.utils.reproducibility import set_seed

DATA_DIR = ROOT / "data" / "forecast"
CKPT_PATH = ROOT / "checkpoints" / "glam_forecast.pt"
REPORTS = ROOT / "reports" / "forecast"
FIG_DIR = ROOT / "paper" / "JAMA Ophthalmology"
CKPT_PATH.parent.mkdir(exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_VF_POINTS = 54
N_CLINICAL = 5
MAX_VISITS = 3
BATCH_SIZE = 64
NLL_WEIGHT = 0.0
PATIENCE = 20
MAX_EPOCHS = 80
LR = 1e-4
STRUCT_DIM = 64
THRESHOLD = -1.0


class DummyBackbone(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.out_dim, device=x.device)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(device: torch.device) -> GLAMModel:
    backbone = DummyBackbone(out_dim=STRUCT_DIM)
    vf_enc = VisualFieldEncoder(n_points=N_VF_POINTS, hidden_dim=256)
    temporal = BidirectionalLSTM(input_dim=256, hidden_dim=256, num_layers=2, dropout=0.3)
    fusion = AttentionFusion(
        structural_dim=STRUCT_DIM,
        functional_dim=temporal.out_dim,
        clinical_dim=128,
        fused_dim=512,
    )
    clin_emb = nn.Sequential(
        nn.Linear(N_CLINICAL, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.GELU(),
    )
    head = ProgressionHead(input_dim=512, dropout=0.3)
    return GLAMModel(backbone, vf_enc, temporal, fusion, clin_emb, head).to(device)


def collect_outputs(model, loader, device):
    model.eval()
    mp, mt, vp, vt, lv, att = [], [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            out = model(
                b["image"].to(device),
                b["vf_series"].to(device),
                b["clinical"].to(device),
                b["vf_length"].to(device),
            )
            mp.append(out["md_pred"].squeeze(-1).cpu().numpy())
            mt.append(b["md_rate"].numpy())
            vp.append(out["vfi_pred"].squeeze(-1).cpu().numpy())
            vt.append(b["vfi_rate"].numpy())
            lv.append(out["md_log_var"].squeeze(-1).cpu().numpy())
            att.append(out["modality_weights"].cpu().numpy())
    return (
        np.concatenate(mp),
        np.concatenate(mt),
        np.concatenate(vp),
        np.concatenate(vt),
        np.concatenate(lv),
        np.concatenate(att, axis=0),
    )


TD_MIN = -35.0


def load_td(pid: str) -> np.ndarray:
    """Observation-window TD maps in decibels, shape (3, 54)."""
    arr = np.load(DATA_DIR / "processed" / f"vf_{pid}.npy")
    return arr * (-TD_MIN) + TD_MIN


FEATURE_NAMES = [
    "age_baseline",
    "gender",
    "input_span_years",
    "md_v1",
    "md_v2",
    "md_v3",
    "delta_md_21",
    "delta_md_32",
    "md_rate_early_ols",
    "td_sd_last",
    "td_p10_last",
    "td_p90_last",
    "td_worst10_last",
]


def build_tabular_features(patient_ids: list[str], patients: pl.DataFrame) -> np.ndarray:
    """Engineered observation-window features giving tabular models the same
    longitudinal information GLAM receives."""
    cols = ["age_baseline", "gender", "input_span_years", "md_rate_early_ols"]
    lookup = {
        row["patient_id"]: row
        for row in patients.select(["patient_id", *cols]).iter_rows(named=True)
    }
    rows = []
    for pid in patient_ids:
        meta = lookup[pid]
        td = load_td(pid)
        md = td.mean(axis=1)
        last = np.sort(td[-1])
        rows.append(
            [
                meta["age_baseline"],
                meta["gender"],
                meta["input_span_years"],
                md[0],
                md[1],
                md[2],
                md[1] - md[0],
                md[2] - md[1],
                meta["md_rate_early_ols"],
                float(td[-1].std()),
                float(np.percentile(td[-1], 10)),
                float(np.percentile(td[-1], 90)),
                float(last[:10].mean()),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def bootstrap_auc(y: np.ndarray, scores: np.ndarray, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], scores[idx]))
    arr = np.asarray(vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """scores: higher = more likely fast (use -md_pred). Returns (thresh_on_score, sens, spec)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])


def save_figures(md_pred, md_true, scores, y_fast, auc, md_log_var, out_dir: Path):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    err = np.abs(md_pred - md_true)
    sc = ax.scatter(md_true, md_pred, c=err, cmap="viridis_r", s=18, alpha=0.8)
    lims = [
        min(md_true.min(), md_pred.min()) - 0.2,
        max(md_true.max(), md_pred.max()) + 0.2,
    ]
    ax.plot(lims, lims, "r--", lw=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True subsequent MD rate (dB/y)")
    ax.set_ylabel("Predicted subsequent MD rate (dB/y)")
    ax.set_title("Predicted vs true subsequent MD rate")
    fig.colorbar(sc, ax=ax, label="|error| (dB/y)")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig2.png", dpi=220)
    fig.savefig(REPORTS / "fig2_pred_vs_true.png", dpi=220)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_fast, scores)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(fpr, tpr, lw=2, label=f"GLAM AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Fast-progressor detection (subsequent MD < −1 dB/y)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig3.png", dpi=220)
    fig.savefig(REPORTS / "fig3_roc.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    qs = np.quantile(md_true, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    maes, ns, labels = [], [], []
    for i in range(5):
        if i < 4:
            mask = (md_true >= qs[i]) & (md_true < qs[i + 1])
        else:
            mask = md_true >= qs[i]
        maes.append(float(np.mean(np.abs(md_pred[mask] - md_true[mask]))) if mask.any() else 0)
        ns.append(int(mask.sum()))
        labels.append(f"Q{i+1}\nn={ns[-1]}")
    ax.bar(range(5), maes, color="#4C72B0")
    ax.set_xticks(range(5), labels)
    ax.set_ylabel("MAE (dB/y)")
    ax.set_title("Test MAE by subsequent MD-rate quintile")
    fig.tight_layout()
    fig.savefig(out_dir / "SuppFig1.png", dpi=220)
    fig.savefig(REPORTS / "efig2_mae_quintile.png", dpi=220)
    plt.close(fig)

    sigma = np.exp(0.5 * np.clip(md_log_var, -8, 8))
    abs_err = np.abs(md_pred - md_true)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(sigma, abs_err, s=16, alpha=0.7)
    ax.set_xlabel("Predicted SD (dB/y)")
    ax.set_ylabel("Absolute error (dB/y)")
    ax.set_title("Aleatoric uncertainty vs absolute error")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig4.png", dpi=220)
    fig.savefig(REPORTS / "fig4_uncertainty.png", dpi=220)
    plt.close(fig)


def main() -> None:
    set_seed(42)
    device = pick_device()
    print(f"Device: {device}")

    train_ds = GlaucomaDataset(
        DATA_DIR, "train", n_clinical_features=N_CLINICAL, max_visits=MAX_VISITS
    )
    val_ds = GlaucomaDataset(
        DATA_DIR, "val", n_clinical_features=N_CLINICAL, max_visits=MAX_VISITS
    )
    test_ds = GlaucomaDataset(
        DATA_DIR, "test", n_clinical_features=N_CLINICAL, max_visits=MAX_VISITS
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train {len(train_ds):,}  Val {len(val_ds):,}  Test {len(test_ds):,}")

    patients = pl.read_parquet(DATA_DIR / "processed" / "patients.parquet")

    model = build_model(device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = WeightedDualRegressionLoss(
        md_weight=1.0,
        vfi_weight=0.8,
        fast_progressor_threshold_md=THRESHOLD,
        fast_progressor_boost=2.5,
    )
    nll_crit = NegativeLogLikelihoodLoss(md_weight=1.0, vfi_weight=0.8)
    nll_weight = NLL_WEIGHT
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )

    def batch_loss(out, b):
        huber = criterion(
            out["md_pred"],
            out["vfi_pred"],
            b["md_rate"].to(device),
            b["vfi_rate"].to(device),
        )
        nll = nll_crit(
            out["md_pred"],
            out["md_log_var"],
            out["vfi_pred"],
            out["vfi_log_var"],
            b["md_rate"].to(device),
            b["vfi_rate"].to(device),
        )
        total = huber["loss"] + nll_weight * nll["loss"]
        return total

    best_val = float("inf")
    no_improve = 0
    history = {"tl": [], "vl": [], "md_mae": []}
    skip_train = CKPT_PATH.exists() and "--eval-only" in sys.argv

    print(f"\n{'Ep':>4} {'TrainLoss':>10} {'ValLoss':>9} {'MD-MAE':>8}")
    print("-" * 36)
    if skip_train:
        print("Loading existing checkpoint (eval-only)")
    else:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            tl_list = []
            for b in train_dl:
                optimizer.zero_grad()
                out = model(
                    b["image"].to(device),
                    b["vf_series"].to(device),
                    b["clinical"].to(device),
                    b["vf_length"].to(device),
                )
                loss = batch_loss(out, b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tl_list.append(loss.item())

            vl_list = []
            model.eval()
            with torch.no_grad():
                for b in val_dl:
                    out = model(
                        b["image"].to(device),
                        b["vf_series"].to(device),
                        b["clinical"].to(device),
                        b["vf_length"].to(device),
                    )
                    vl_list.append(batch_loss(out, b).item())
            md_pred_v, md_true_v, *_ = collect_outputs(model, val_dl, device)
            scheduler.step()
            tl = float(np.mean(tl_list))
            vl = float(np.mean(vl_list))
            md_mae = float(np.mean(np.abs(md_pred_v - md_true_v)))
            history["tl"].append(tl)
            history["vl"].append(vl)
            history["md_mae"].append(md_mae)
            if vl < best_val:
                best_val = vl
                no_improve = 0
                torch.save(
                    {"model_state": model.state_dict(), "epoch": epoch, "val_loss": vl},
                    CKPT_PATH,
                )
            else:
                no_improve += 1
            print(f"{epoch:>4} {tl:>10.4f} {vl:>9.4f} {md_mae:>8.4f}", flush=True)
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    md_pred, md_true, vfi_pred, vfi_true, md_log_var, att = collect_outputs(
        model, test_dl, device
    )
    glam_metrics = compute_all_metrics(md_pred, md_true, vfi_pred, vfi_true)

    id_to_row = {
        pid: i for i, pid in enumerate(patients["patient_id"].to_list())
    }
    early_all = patients["md_rate_early_ols"].to_numpy()
    md_base_all = patients["md_baseline"].to_numpy()
    early_ols = np.array([early_all[id_to_row[pid]] for pid in test_ds.patient_ids])
    baseline_md = np.array([md_base_all[id_to_row[pid]] for pid in test_ds.patient_ids]).reshape(-1, 1)
    train_ids = (DATA_DIR / "splits" / "train_ids.txt").read_text().strip().splitlines()
    train_df = patients.filter(pl.col("patient_id").is_in(train_ids))
    naive = np.full_like(md_true, float(train_df["md_rate_db_year"].mean()))
    ridge = Ridge(alpha=1.0)
    ridge.fit(
        train_df["md_baseline"].to_numpy().reshape(-1, 1),
        train_df["md_rate_db_year"].to_numpy(),
    )
    ridge_pred = ridge.predict(baseline_md)

    # Stronger comparators: same observation-window longitudinal information as GLAM
    y_train = train_df["md_rate_db_year"].to_numpy()
    x_train = build_tabular_features(train_df["patient_id"].to_list(), patients)
    x_test = build_tabular_features(list(test_ds.patient_ids), patients)

    ridge_full = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge_full.fit(x_train, y_train)
    ridge_full_pred = ridge_full.predict(x_test)

    gbm = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    gbm.fit(x_train, y_train)
    gbm_pred = gbm.predict(x_test)

    def pack(name, pred):
        m = compute_all_metrics(pred, md_true, np.zeros_like(pred), vfi_true)
        out = {"name": name, **{k: float(v) if np.isfinite(v) else None for k, v in m.items()}}
        y = (md_true < THRESHOLD).astype(int)
        if y.min() != y.max():
            lo, hi = bootstrap_auc(y, -np.asarray(pred, dtype=np.float64))
            out["auc_ci95"] = [lo, hi]
        return out

    y_fast = (md_true < THRESHOLD).astype(int)
    scores = -md_pred
    auc = float(roc_auc_score(y_fast, scores)) if y_fast.min() != y_fast.max() else float("nan")
    auc_lo, auc_hi = bootstrap_auc(y_fast, scores)

    val_pred, val_true, *_ = collect_outputs(model, val_dl, device)
    y_val = (val_true < THRESHOLD).astype(int)
    score_thr, _, _ = youden_threshold(y_val, -val_pred)
    md_thr = -score_thr
    is_fast_pred = md_pred < md_thr
    tp = int(((is_fast_pred == 1) & (y_fast == 1)).sum())
    tn = int(((is_fast_pred == 0) & (y_fast == 0)).sum())
    fp = int(((is_fast_pred == 1) & (y_fast == 0)).sum())
    fn = int(((is_fast_pred == 0) & (y_fast == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0

    sigma = np.exp(0.5 * np.clip(md_log_var, -8, 8))
    z = 1.6448536269514722
    coverage90 = float(np.mean(np.abs(md_pred - md_true) <= z * sigma))
    abs_err = np.abs(md_pred - md_true)
    rho = float(np.corrcoef(sigma, abs_err)[0, 1]) if len(sigma) > 2 else float("nan")

    def split_stats(ids):
        d = patients.filter(pl.col("patient_id").is_in(ids))
        rates = d["md_rate_db_year"].to_numpy()
        return {
            "n": len(d),
            "n_fast": int((rates < THRESHOLD).sum()),
            "md_rate_mean": float(rates.mean()),
            "md_rate_std": float(rates.std()),
            "n_total_visits_mean": float(d["n_total_visits"].mean()),
            "n_total_visits_std": float(d["n_total_visits"].std()),
            "n_future_visits_mean": float(d["n_future_visits"].mean()),
            "input_span_median": float(d["input_span_years"].median()),
            "future_span_median": float(d["future_span_years"].median()),
            "future_span_q25": float(d["future_span_years"].quantile(0.25)),
            "future_span_q75": float(d["future_span_years"].quantile(0.75)),
            "md_baseline_mean": float(d["md_baseline"].mean()),
            "md_baseline_std": float(d["md_baseline"].std()),
        }

    val_ids = (DATA_DIR / "splits" / "val_ids.txt").read_text().strip().splitlines()
    test_ids = (DATA_DIR / "splits" / "test_ids.txt").read_text().strip().splitlines()

    results = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_fast_test": int(y_fast.sum()),
        "glam": {k: float(v) if np.isfinite(v) else None for k, v in glam_metrics.items()},
        "glam_auc": auc,
        "glam_auc_ci95": [auc_lo, auc_hi],
        "youden_md_threshold": float(md_thr),
        "youden_sensitivity": sens,
        "youden_specificity": spec,
        "youden_ppv": ppv,
        "youden_tp": tp,
        "youden_tn": tn,
        "youden_fp": fp,
        "youden_fn": fn,
        "uncertainty_coverage90": coverage90,
        "uncertainty_corr": rho,
        "attention_mean": {
            "structural": float(att[:, 0].mean()),
            "functional": float(att[:, 1].mean()),
            "clinical": float(att[:, 2].mean()),
        },
        "baselines": [
            pack("naive_mean", naive),
            pack("ridge_baseline_md", ridge_pred),
            pack("ols_early_3visits", early_ols),
            pack("ridge_full_features", ridge_full_pred),
            pack("gbm_full_features", gbm_pred),
        ],
        "feature_names": FEATURE_NAMES,
        "splits": {
            "all": split_stats(train_ids + val_ids + test_ids),
            "train": split_stats(train_ids),
            "val": split_stats(val_ids),
            "test": split_stats(test_ids),
        },
        "best_epoch": ckpt.get("epoch"),
        "best_val_loss": float(ckpt.get("val_loss", best_val)),
    }
    (REPORTS / "forecast_metrics.json").write_text(json.dumps(results, indent=2))
    json.dump(history, open(REPORTS / "training_history.json", "w"))
    print(json.dumps(results, indent=2))
    save_figures(md_pred, md_true, scores, y_fast, auc, md_log_var, FIG_DIR)
    print(f"Wrote metrics to {REPORTS / 'forecast_metrics.json'}")
    print(f"Checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()
