# Tech Stack Guide — GLAM Project

## Quick Reference

| Category | Tool | Why |
|----------|------|-----|
| Package Manager | **uv** | 10–100× faster than pip, Rust-based |
| Deep Learning | **PyTorch 2.4+** | Industry standard for research |
| Training Framework | **PyTorch Lightning 2.4+** | Structured training, no boilerplate |
| Data Processing | **Polars 1.0+** | Fast time-series groupby, modern API |
| Config Management | **Hydra** | Manage 50+ hyperparameters cleanly |
| Experiment Tracking | **Weights & Biases** | Curves, artifact management |
| Hyperparameter Search | **Optuna** | Bayesian optimization |
| Interpretability | **SHAP** | Feature importance for clinical insights |
| Survival Analysis | **Lifelines** | Time-to-blindness estimation |
| Claude API | **anthropic>=0.49** | Idea validation + knowledge base queries |

---

## Installation

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project environment and install all dependencies
cd glam/
uv sync

# For development tools
uv sync --extra dev

# Activate environment
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows

# Run idea validator
cp .env.example .env
# Edit .env → add ANTHROPIC_API_KEY
python validate_idea.py

# Train model
python scripts/train.py

# Train with overrides
python scripts/train.py training.learning_rate=1e-4 data.batch_size=16
```

---

## Polars vs Pandas for This Project

### Use Polars For:
```python
import polars as pl

# Fast temporal groupby (visit alignment)
visits_df.group_by("patient_id").agg(
    pl.col("visit_date").sort().alias("visits"),
    pl.col("md_db").sort_by("visit_date").alias("md_series"),
)

# Temporal resampling (important: Polars uses μs, not ns like Pandas)
df.with_columns(
    pl.col("visit_date").dt.total_days().alias("days_from_baseline")
)

# Fast parquet I/O (much faster than Pandas)
pl.read_parquet("data/processed/patients.parquet")
pl.write_parquet("output.parquet", compression="snappy")
```

### Use Pandas For:
```python
import pandas as pd
# For compatibility with sklearn, SHAP, and legacy code
# Convert: df.to_pandas() / pl.from_pandas(pd_df)
```

---

## PyTorch Lightning Structure

```python
class GLAMTrainer(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # Saves to checkpoint

    def training_step(self, batch, batch_idx):
        # Forward + loss + logging
        ...

    def validation_step(self, batch, batch_idx):
        # Eval + metric logging
        ...

    def configure_optimizers(self):
        # Optimizer + scheduler
        return {"optimizer": ..., "lr_scheduler": ...}
```

Benefits:
- Automatic GPU/multi-GPU support
- Built-in checkpoint management
- No manual `optimizer.zero_grad()`, `loss.backward()`, etc.

---

## Hydra Config System

```python
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Access config
    lr = cfg.training.learning_rate
    model = hydra.utils.instantiate(cfg.model)  # Auto-builds model!
```

### Command-Line Overrides
```bash
# Change learning rate
python scripts/train.py training.learning_rate=5e-4

# Use different model config
python scripts/train.py model=transformer_gru

# Multi-run sweep
python scripts/train.py --multirun training.learning_rate=1e-3,1e-4,1e-5
```

---

## Weights & Biases Integration

```python
import wandb

# In training script
if cfg.experiment.use_wandb:
    wandb.init(project="glam", config=OmegaConf.to_container(cfg))

# Log metrics
wandb.log({"val/md_mae": 0.45, "val/vfi_mae": 1.2, "epoch": 10})

# Log model artifacts
wandb.save("checkpoints/best_model.ckpt")
```

### Setup
```bash
wandb login  # Enter your API key once
# Set WANDB_API_KEY in .env
```

---

## Optuna Hyperparameter Search

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])

    # Train with these hyperparams, return validation MAE
    return val_mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```

---

## SHAP for Model Interpretability

```python
import shap
import numpy as np

# For clinical features (tabular)
explainer = shap.TreeExplainer(rf_model)  # or DeepExplainer for DL
shap_values = explainer.shap_values(X_test)

# Top features for fast progressor prediction
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=CLINICAL_FEATURE_NAMES,
    plot_type="bar",
)

# Per-patient explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    feature_names=CLINICAL_FEATURE_NAMES,
)
```

---

## Claude API Usage in This Project

### Idea Validation
```bash
python validate_idea.py  # Full project analysis with adaptive thinking
```

### Potential Research Use Cases
```python
import anthropic

client = anthropic.Anthropic()

# Summarize a paper abstract
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"Summarize this ophthalmology paper for a DL researcher:\n\n{abstract}"
    }]
)

# Generate synthetic patient summaries for testing
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": "Generate 5 realistic clinical vignettes for highly myopic glaucoma patients with different progression rates. Include: age, IOP history, axial length, baseline MD, follow-up duration, and outcome."
    }]
)
```

---

## GPU Requirements

| Scenario | Min GPU | Recommended |
|----------|---------|-------------|
| Debugging (batch_size=4) | 4GB VRAM | Any RTX 3060 |
| Training ConvNeXt-S | 8GB VRAM | RTX 3080 |
| Full training (batch_size=32) | 16GB VRAM | RTX 4080 / A100 |
| Hyperparameter sweep | 40GB VRAM | A100 / H100 |

### Cloud Options
- **Google Colab Pro+**: A100 40GB (most cost-effective for this scale)
- **Lambda Labs**: H100 at ~$2/hr
- **AWS SageMaker**: For production pipelines
- **Vast.ai**: Cheapest GPU rental for research

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
