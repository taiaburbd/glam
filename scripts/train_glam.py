"""
Train GLAM on real UWHVF data. Run with: uv run python scripts/train_glam.py
"""
import sys, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.backbones import ConvNeXtBackbone, VisualFieldEncoder
from src.models.temporal import BidirectionalLSTM
from src.models.fusion import AttentionFusion
from src.models.heads import ProgressionHead, GLAMModel
from src.data.loaders import GlaucomaDataset
from src.losses.progression import WeightedDualRegressionLoss
from src.utils.metrics import compute_all_metrics
from src.utils.reproducibility import set_seed

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "glam_real.pt"
CKPT_PATH.parent.mkdir(exist_ok=True)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

N_VF_POINTS = 54
N_CLINICAL  = 5
MAX_VISITS  = 12
BATCH_SIZE  = 64
MAX_EPOCHS  = 60
LR          = 3e-4
PATIENCE    = 12

train_ds = GlaucomaDataset(ROOT / "data", "train", n_clinical_features=N_CLINICAL, max_visits=MAX_VISITS)
val_ds   = GlaucomaDataset(ROOT / "data", "val",   n_clinical_features=N_CLINICAL, max_visits=MAX_VISITS)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

# No fundus images in UWHVF — use a lightweight placeholder (64-dim zeros → constant)
# This is honest: the model is VF-functional + clinical only.
STRUCT_DIM = 64

class DummyBackbone(nn.Module):
    """Placeholder structural branch — outputs zeros (no fundus images in UWHVF)."""
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.out_dim, device=x.device)

backbone  = DummyBackbone(out_dim=STRUCT_DIM)
vf_enc    = VisualFieldEncoder(n_points=N_VF_POINTS, hidden_dim=256)
temporal  = BidirectionalLSTM(input_dim=256, hidden_dim=256, num_layers=2, dropout=0.3)
fusion    = AttentionFusion(structural_dim=STRUCT_DIM, functional_dim=temporal.out_dim,
                            clinical_dim=128, fused_dim=512)
clin_emb  = nn.Sequential(nn.Linear(N_CLINICAL,64), nn.LayerNorm(64),  nn.GELU(),
                           nn.Linear(64, 128),        nn.LayerNorm(128), nn.GELU())
head      = ProgressionHead(input_dim=512, dropout=0.3)
model     = GLAMModel(backbone, vf_enc, temporal, fusion, clin_emb, head).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = WeightedDualRegressionLoss(
    md_weight=1.0, vfi_weight=0.8,
    fast_progressor_threshold_md=-1.0, fast_progressor_boost=2.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

best_val = float("inf"); no_improve = 0
history  = {"tl": [], "vl": [], "md_mae": [], "vfi_mae": []}

print(f"\n{'Ep':>4} {'TrainLoss':>10} {'ValLoss':>9} {'MD-MAE':>8} {'VFI-MAE':>9}")
print("-" * 46)

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tl_list = []
    for b in train_dl:
        optimizer.zero_grad()
        out  = model(b["image"].to(DEVICE), b["vf_series"].to(DEVICE),
                     b["clinical"].to(DEVICE), b["vf_length"].to(DEVICE))
        loss = criterion(out["md_pred"], out["vfi_pred"],
                         b["md_rate"].to(DEVICE), b["vfi_rate"].to(DEVICE))
        loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl_list.append(loss["loss"].item())

    model.eval()
    vl_list, mp, mt, vp, vt = [], [], [], [], []
    with torch.no_grad():
        for b in val_dl:
            out  = model(b["image"].to(DEVICE), b["vf_series"].to(DEVICE),
                         b["clinical"].to(DEVICE), b["vf_length"].to(DEVICE))
            loss = criterion(out["md_pred"], out["vfi_pred"],
                             b["md_rate"].to(DEVICE), b["vfi_rate"].to(DEVICE))
            vl_list.append(loss["loss"].item())
            mp.append(out["md_pred"] .squeeze().cpu().numpy())
            mt.append(b["md_rate"]   .numpy())
            vp.append(out["vfi_pred"].squeeze().cpu().numpy())
            vt.append(b["vfi_rate"]  .numpy())

    scheduler.step()
    tl      = np.mean(tl_list)
    vl      = np.mean(vl_list)
    md_mae  = np.mean(np.abs(np.concatenate(mp) - np.concatenate(mt)))
    vfi_mae = np.mean(np.abs(np.concatenate(vp) - np.concatenate(vt)))

    history["tl"].append(float(tl)); history["vl"].append(float(vl))
    history["md_mae"].append(float(md_mae)); history["vfi_mae"].append(float(vfi_mae))

    if vl < best_val:
        best_val   = vl
        no_improve = 0
        torch.save({"model_state": model.state_dict(), "epoch": epoch,
                    "val_loss": vl, "val_md_mae": md_mae,
                    "arch": {"N_VF_POINTS": N_VF_POINTS, "N_CLINICAL": N_CLINICAL,
                             "MAX_VISITS": MAX_VISITS}},
                   CKPT_PATH)
    else:
        no_improve += 1

    print(f"{epoch:>4} {tl:>10.4f} {vl:>9.4f} {md_mae:>8.4f} {vfi_mae:>9.4f}", flush=True)
    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\nBest val loss: {best_val:.4f}")
json.dump(history, open(CKPT_PATH.parent / "training_history.json", "w"))
print(f"Checkpoint: {CKPT_PATH}")
