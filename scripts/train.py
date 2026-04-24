"""
GLAM Training Script — PyTorch Lightning + Hydra
Usage: python scripts/train.py [overrides...]
       python scripts/train.py training.learning_rate=1e-4 data.batch_size=16
"""

import hydra
import torch
import lightning as L
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.utils.reproducibility import set_seed
from src.data import create_dataloaders
from src.losses import WeightedDualRegressionLoss, NegativeLogLikelihoodLoss
from src.utils.metrics import compute_all_metrics


class GLAMTrainer(L.LightningModule):
    def __init__(self, model: torch.nn.Module, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        if cfg.loss.type == "nll":
            self.criterion = NegativeLogLikelihoodLoss(
                md_weight=cfg.loss.md_weight,
                vfi_weight=cfg.loss.vfi_weight,
            )
        else:
            self.criterion = WeightedDualRegressionLoss(
                md_weight=cfg.loss.md_weight,
                vfi_weight=cfg.loss.vfi_weight,
                fast_progressor_threshold_md=cfg.loss.fast_progressor_threshold,
                fast_progressor_boost=cfg.loss.fast_progressor_boost,
            )

        self._val_preds: list = []

    def forward(self, batch: dict) -> dict:
        return self.model(
            images=batch["image"],
            vf_series=batch["vf_series"],
            clinical=batch["clinical"],
            vf_lengths=batch["vf_length"],
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        preds = self(batch)
        losses = self.criterion(
            md_pred=preds["md_pred"],
            vfi_pred=preds["vfi_pred"],
            md_target=batch["md_rate"],
            vfi_target=batch["vfi_rate"],
        )
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        preds = self(batch)
        losses = self.criterion(
            md_pred=preds["md_pred"],
            vfi_pred=preds["vfi_pred"],
            md_target=batch["md_rate"],
            vfi_target=batch["vfi_rate"],
        )
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, prog_bar=True)
        self._val_preds.append({
            "md_pred": preds["md_pred"].detach().cpu(),
            "md_true": batch["md_rate"].cpu(),
            "vfi_pred": preds["vfi_pred"].detach().cpu(),
            "vfi_true": batch["vfi_rate"].cpu(),
        })

    def on_validation_epoch_end(self) -> None:
        import numpy as np
        md_pred = torch.cat([p["md_pred"] for p in self._val_preds]).squeeze().numpy()
        md_true = torch.cat([p["md_true"] for p in self._val_preds]).numpy()
        vfi_pred = torch.cat([p["vfi_pred"] for p in self._val_preds]).squeeze().numpy()
        vfi_true = torch.cat([p["vfi_true"] for p in self._val_preds]).numpy()

        metrics = compute_all_metrics(md_pred, md_true, vfi_pred, vfi_true)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        self._val_preds.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.project.seed)

    dataloaders = create_dataloaders(
        data_dir=cfg.data.dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Build model from Hydra config
    model = hydra.utils.instantiate(cfg.model)
    lit_model = GLAMTrainer(model, cfg)

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=cfg.experiment.checkpoint_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=cfg.experiment.save_top_k,
            filename="glam-{epoch:02d}-{val/loss:.4f}",
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if cfg.experiment.use_wandb:
        wandb.init(project=cfg.project.name, config=OmegaConf.to_container(cfg))

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        deterministic=True,
    )

    trainer.fit(
        lit_model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
    )


if __name__ == "__main__":
    main()
