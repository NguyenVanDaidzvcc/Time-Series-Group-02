from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds used by numpy, random and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_processed_dir(cwd: Path | None = None) -> Path:
    """Resolve the processed data directory from notebook or project root."""
    base = Path.cwd() if cwd is None else Path(cwd)
    candidates = [
        base / ".." / "data" / "processed",
        base / "data" / "processed",
        base / "time_series _" / "data" / "processed",
    ]
    for path in candidates:
        if (path / "feature_metadata.json").exists():
            return path.resolve()
    raise FileNotFoundError("Khong tim thay thu muc data/processed. Hay chay feature pipeline truoc.")


def load_processed_artifacts(processed_dir: Path | None = None) -> dict[str, object]:
    """Load metadata, scalers and train/validation/test windows."""
    resolved_dir = resolve_processed_dir() if processed_dir is None else Path(processed_dir).resolve()
    with open(resolved_dir / "feature_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    artifacts = {
        "processed_dir": resolved_dir,
        "project_dir": resolved_dir.parents[1],
        "metadata": metadata,
        "feature_scaler": joblib.load(resolved_dir / "feature_scaler.pkl"),
        "target_scaler": joblib.load(resolved_dir / "target_scaler.pkl"),
    }
    for split, file_name in [
        ("train", "train_windows.npz"),
        ("validation", "validation_windows.npz"),
        ("test", "test_windows.npz"),
    ]:
        npz = np.load(resolved_dir / file_name)
        artifacts[split] = {
            "X": npz["X"].astype(np.float32),
            "y": npz["y"].astype(np.float32),
            "window_start": npz["window_start"],
            "target_start": npz["target_start"],
            "target_end": npz["target_end"],
        }
    return artifacts


def inverse_target(y_scaled: np.ndarray, target_scaler) -> np.ndarray:
    """Inverse-transform scaled target arrays back to original target units."""
    y_scaled = np.asarray(y_scaled)
    y_2d = y_scaled.reshape(-1, 1)
    return target_scaler.inverse_transform(y_2d).reshape(y_scaled.shape)


def regression_metrics(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, target_scaler) -> dict[str, float]:
    """Compute MAE, RMSE and R2 after inverse scaling."""
    y_true = inverse_target(y_true_scaled, target_scaler)
    y_pred = inverse_target(y_pred_scaled, target_scaler)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "rmse": float(mean_squared_error(y_true_flat, y_pred_flat) ** 0.5),
        "r2": float(r2_score(y_true_flat, y_pred_flat)),
    }


def save_json(data: dict, path: Path) -> Path:
    """Save a dictionary as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def summarize_model(
    name: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_seconds: float,
) -> dict:
    """Build the training summary dictionary used by notebooks and reports."""
    return {
        "model": name,
        "train_seconds": float(train_seconds),
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }


def last_step_tabular_features(X_seq: np.ndarray) -> np.ndarray:
    """Use the last input timestamp from a sequence window as tabular features."""
    return X_seq[:, -1, :]


def train_linear_regression(
    artifacts: dict[str, object],
    model_dir: Path,
    history_dir: Path,
) -> tuple[LinearRegression, dict]:
    """Train and save the Linear Regression baseline."""
    target_scaler = artifacts["target_scaler"]
    X_train = last_step_tabular_features(artifacts["train"]["X"])
    X_val = last_step_tabular_features(artifacts["validation"]["X"])
    X_test = last_step_tabular_features(artifacts["test"]["X"])
    y_train = artifacts["train"]["y"]
    y_val = artifacts["validation"]["y"]
    y_test = artifacts["test"]["y"]

    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start_time

    history = summarize_model(
        "linear_regression",
        regression_metrics(y_train, model.predict(X_train), target_scaler),
        regression_metrics(y_val, model.predict(X_val), target_scaler),
        regression_metrics(y_test, model.predict(X_test), target_scaler),
        train_seconds,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "linear_regression.pkl")
    save_json(history, history_dir / "linear_regression_history.json")
    return model, history


def train_xgboost(
    artifacts: dict[str, object],
    model_dir: Path,
    history_dir: Path,
    seed: int = SEED,
) -> tuple[MultiOutputRegressor, dict]:
    """Train and save the XGBoost multi-output model."""
    from xgboost import XGBRegressor

    target_scaler = artifacts["target_scaler"]
    X_train = last_step_tabular_features(artifacts["train"]["X"])
    X_val = last_step_tabular_features(artifacts["validation"]["X"])
    X_test = last_step_tabular_features(artifacts["test"]["X"])
    y_train = artifacts["train"]["y"]
    y_val = artifacts["validation"]["y"]
    y_test = artifacts["test"]["y"]

    xgb_base = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model = MultiOutputRegressor(xgb_base, n_jobs=1)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start_time

    history = summarize_model(
        "xgboost",
        regression_metrics(y_train, model.predict(X_train), target_scaler),
        regression_metrics(y_val, model.predict(X_val), target_scaler),
        regression_metrics(y_test, model.predict(X_test), target_scaler),
        train_seconds,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "xgboost_multioutput.pkl")
    save_json(history, history_dir / "xgboost_history.json")
    return model, history


class ITransformer(nn.Module):
    """Compact iTransformer-style encoder for multivariate time-series forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.variable_embedding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x) + self.variable_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


def make_sequence_loaders(
    artifacts: dict[str, object],
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders from sequence windows."""
    train_dataset = TensorDataset(torch.from_numpy(artifacts["train"]["X"]), torch.from_numpy(artifacts["train"]["y"]))
    val_dataset = TensorDataset(
        torch.from_numpy(artifacts["validation"]["X"]),
        torch.from_numpy(artifacts["validation"]["y"]),
    )
    test_dataset = TensorDataset(torch.from_numpy(artifacts["test"]["X"]), torch.from_numpy(artifacts["test"]["y"]))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer=None,
    device: torch.device | str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one training or evaluation epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_samples = 0
    preds = []
    targets = []

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size_now = X_batch.size(0)
            total_loss += loss.item() * batch_size_now
            total_samples += batch_size_now
            preds.append(y_pred.detach().cpu().numpy())
            targets.append(y_batch.detach().cpu().numpy())

    return total_loss / total_samples, np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def train_itransformer(
    artifacts: dict[str, object],
    model_dir: Path,
    history_dir: Path,
    batch_size: int = 256,
    epochs: int = 10,
    patience: int = 3,
    learning_rate: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    device: torch.device | None = None,
) -> tuple[ITransformer, dict]:
    """Train and save the iTransformer model."""
    metadata = artifacts["metadata"]
    target_scaler = artifacts["target_scaler"]
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    train_loader, val_loader, test_loader = make_sequence_loaders(artifacts, batch_size)

    model = ITransformer(
        seq_len=metadata["seq_len"],
        pred_len=metadata["pred_len"],
        n_features=len(metadata["feature_cols"]),
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(resolved_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "model": "itransformer",
        "config": {
            "seq_len": metadata["seq_len"],
            "pred_len": metadata["pred_len"],
            "n_features": len(metadata["feature_cols"]),
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "d_model": d_model,
            "n_heads": n_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "device": str(resolved_device),
        },
        "epochs": [],
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, resolved_device)
        val_loss, val_pred, val_target = run_epoch(model, val_loader, criterion, None, resolved_device)
        scheduler.step(val_loss)
        val_metrics = regression_metrics(val_target, val_pred, target_scaler)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history["epochs"].append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    train_seconds = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    train_loss_final, train_pred, train_target = run_epoch(model, train_loader, criterion, None, resolved_device)
    val_loss_final, val_pred, val_target = run_epoch(model, val_loader, criterion, None, resolved_device)
    test_loss_final, test_pred, test_target = run_epoch(model, test_loader, criterion, None, resolved_device)

    history["train_seconds"] = float(train_seconds)
    history["best_val_loss"] = float(best_val_loss)
    history["final"] = {
        "train_loss": float(train_loss_final),
        "val_loss": float(val_loss_final),
        "test_loss": float(test_loss_final),
        "train": regression_metrics(train_target, train_pred, target_scaler),
        "validation": regression_metrics(val_target, val_pred, target_scaler),
        "test": regression_metrics(test_target, test_pred, target_scaler),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "model_config": history["config"],
        },
        model_dir / "itransformer.pt",
    )
    save_json(history, history_dir / "itransformer_history.json")
    return model, history


def build_summary_dataframe(histories: dict[str, dict]) -> pd.DataFrame:
    """Build a comparable model summary dataframe from training histories."""
    rows = []
    for model_name, history in histories.items():
        final = history.get("final", history)
        rows.append(
            {
                "model": model_name,
                "train_seconds": history["train_seconds"],
                **{f"validation_{key}": value for key, value in final["validation"].items()},
                **{f"test_{key}": value for key, value in final["test"].items()},
            }
        )
    return pd.DataFrame(rows).sort_values("validation_rmse")


def train_all_models(processed_dir: Path | None = None, seed: int = SEED) -> dict[str, object]:
    """Train Linear Regression, XGBoost and iTransformer from processed artifacts."""
    set_seed(seed)
    artifacts = load_processed_artifacts(processed_dir)
    project_dir = artifacts["project_dir"]
    model_dir = project_dir / "results" / "models"
    history_dir = project_dir / "results" / "training_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    _, linear_history = train_linear_regression(artifacts, model_dir, history_dir)
    _, xgb_history = train_xgboost(artifacts, model_dir, history_dir, seed)
    _, itransformer_history = train_itransformer(artifacts, model_dir, history_dir)

    histories = {
        "linear_regression": linear_history,
        "xgboost": xgb_history,
        "itransformer": itransformer_history,
    }
    summary_df = build_summary_dataframe(histories)
    summary_df.to_csv(history_dir / "model_summary.csv", index=False)
    save_json(histories, history_dir / "all_training_history.json")
    return {"histories": histories, "summary": summary_df, "model_dir": model_dir, "history_dir": history_dir}
