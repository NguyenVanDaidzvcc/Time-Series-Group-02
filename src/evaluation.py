from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

try:
    from .models import ITransformer, inverse_target, last_step_tabular_features, load_processed_artifacts
except ImportError:
    from models import ITransformer, inverse_target, last_step_tabular_features, load_processed_artifacts


sns.set_theme(style="whitegrid", palette="Set2")


def load_torch_checkpoint(path: Path, device: torch.device):
    """Load a torch checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_trained_models(model_dir: Path, metadata: dict, device: torch.device | None = None) -> dict[str, object]:
    """Load Linear Regression, XGBoost and iTransformer artifacts."""
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    linear_model = joblib.load(model_dir / "linear_regression.pkl")
    xgb_model = joblib.load(model_dir / "xgboost_multioutput.pkl")

    checkpoint = load_torch_checkpoint(model_dir / "itransformer.pt", resolved_device)
    config = checkpoint.get("model_config", {})
    itransformer_model = ITransformer(
        seq_len=config.get("seq_len", metadata["seq_len"]),
        pred_len=config.get("pred_len", metadata["pred_len"]),
        n_features=config.get("n_features", len(metadata["feature_cols"])),
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
    ).to(resolved_device)
    itransformer_model.load_state_dict(checkpoint["model_state_dict"])
    itransformer_model.eval()
    return {
        "Linear Regression": linear_model,
        "XGBoost": xgb_model,
        "iTransformer": itransformer_model,
        "_device": resolved_device,
    }


def predict_itransformer(
    model: ITransformer,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict with iTransformer over a sequence window array."""
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (X_batch,) in loader:
            preds.append(model(X_batch.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_all_models(models: dict[str, object], X_test_seq: np.ndarray) -> dict[str, np.ndarray]:
    """Generate scaled predictions for all trained models."""
    X_test_tab = last_step_tabular_features(X_test_seq)
    return {
        "Linear Regression": models["Linear Regression"].predict(X_test_tab),
        "XGBoost": models["XGBoost"].predict(X_test_tab),
        "iTransformer": predict_itransformer(models["iTransformer"], X_test_seq, models["_device"]),
    }


def regression_metrics_original_scale(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE and R2 on original-scale target values."""
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)
    nonzero_mask = y_true_flat != 0
    return {
        "MAE": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "RMSE": float(mean_squared_error(y_true_flat, y_pred_flat) ** 0.5),
        "MAPE (%)": float(
            np.mean(np.abs((y_true_flat[nonzero_mask] - y_pred_flat[nonzero_mask]) / y_true_flat[nonzero_mask])) * 100
        ),
        "R2": float(r2_score(y_true_flat, y_pred_flat)),
    }


def compare_models(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a metrics table sorted by RMSE."""
    rows = []
    for model_name, y_pred in predictions.items():
        row = {"Model": model_name}
        row.update(regression_metrics_original_scale(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def horizon_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute MAE for each forecast horizon."""
    return np.mean(np.abs(y_true - y_pred), axis=0)


def build_horizon_metrics(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build a dataframe with MAE for each forecast horizon and model."""
    rows = []
    for model_name, y_pred in predictions.items():
        for horizon_idx, mae in enumerate(horizon_mae(y_true, y_pred), start=1):
            rows.append({"Model": model_name, "Horizon": horizon_idx, "MAE": float(mae)})
    return pd.DataFrame(rows)


def build_prediction_dataframe(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_start: np.ndarray,
) -> pd.DataFrame:
    """Create a long dataframe of original-scale actuals and predictions."""
    rows = []
    target_start = pd.to_datetime(target_start)
    pred_len = y_true.shape[1]
    for i in range(len(y_true)):
        for horizon_idx in range(pred_len):
            row = {
                "target_start": target_start[i],
                "horizon": horizon_idx + 1,
                "actual_traffic_volume": y_true[i, horizon_idx],
            }
            for model_name, y_pred in predictions.items():
                row[f"{model_name}_prediction"] = y_pred[i, horizon_idx]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: Path | None = None):
    """Plot MAE, RMSE, MAPE and R2 bars for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric in zip(axes.ravel(), ["MAE", "RMSE", "MAPE (%)", "R2"]):
        sns.barplot(data=metrics_df, x="Model", y=metric, ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("So sanh metrics tren test set", fontsize=16, y=1.02)
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_horizon_mae(horizon_metrics_df: pd.DataFrame, pred_len: int, output_path: Path | None = None):
    """Plot forecast-horizon MAE for each model."""
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=horizon_metrics_df, x="Horizon", y="MAE", hue="Model", marker="o", ax=ax)
    ax.set_title("MAE theo tung buoc du bao 1-24 gio")
    ax.set_xlabel("Forecast horizon (gio)")
    ax.set_ylabel("MAE traffic_volume")
    ax.set_xticks(range(1, pred_len + 1))
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_actual_vs_prediction_horizon_1(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_start: np.ndarray,
    output_path: Path | None = None,
    plot_points: int = 24 * 7,
):
    """Plot actual vs predicted values for the +1 hour horizon."""
    points = min(plot_points, len(y_true))
    time_index = pd.to_datetime(target_start[:points])
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(time_index, y_true[:points, 0], label="Actual", color="black", linewidth=2)
    for model_name, y_pred in predictions.items():
        ax.plot(time_index, y_pred[:points, 0], label=model_name, alpha=0.85)
    ax.set_title("Actual vs Prediction cho horizon +1 gio")
    ax.set_xlabel("Thoi gian")
    ax.set_ylabel("traffic_volume")
    ax.legend()
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path | None = None,
    scatter_sample: int = 5000,
    seed: int = 42,
):
    """Plot sampled actual-vs-predicted scatter charts for all models."""
    sample_size = min(scatter_sample, y_true.size)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(y_true.size, size=sample_size, replace=False)
    actual_flat = y_true.reshape(-1)[sample_idx]
    min_value = min(actual_flat.min(), *(pred.reshape(-1)[sample_idx].min() for pred in predictions.values()))
    max_value = max(actual_flat.max(), *(pred.reshape(-1)[sample_idx].max() for pred in predictions.values()))

    fig, axes = plt.subplots(1, len(predictions), figsize=(16, 5), sharex=True, sharey=True)
    if len(predictions) == 1:
        axes = [axes]
    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        pred_flat = y_pred.reshape(-1)[sample_idx]
        ax.scatter(actual_flat, pred_flat, s=8, alpha=0.25)
        ax.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    fig.suptitle("Scatter Actual vs Predicted tren test set", fontsize=16, y=1.02)
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def run_evaluation(processed_dir: Path | None = None, save_outputs: bool = True) -> dict[str, object]:
    """Run the full test-set evaluation pipeline."""
    artifacts = load_processed_artifacts(processed_dir)
    project_dir = artifacts["project_dir"]
    model_dir = project_dir / "results" / "models"
    evaluation_dir = project_dir / "results" / "evaluation"
    figure_dir = project_dir / "figures" / "evaluation"
    if save_outputs:
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

    y_test_scaled = artifacts["test"]["y"]
    X_test_seq = artifacts["test"]["X"]
    y_test = inverse_target(y_test_scaled, artifacts["target_scaler"])
    loaded_models = load_trained_models(model_dir, artifacts["metadata"])
    predictions_scaled = predict_all_models(loaded_models, X_test_seq)
    predictions = {
        model_name: inverse_target(pred, artifacts["target_scaler"])
        for model_name, pred in predictions_scaled.items()
    }
    metrics_df = compare_models(y_test, predictions)
    horizon_metrics_df = build_horizon_metrics(y_test, predictions)
    prediction_df = build_prediction_dataframe(y_test, predictions, artifacts["test"]["target_start"])

    if save_outputs:
        metrics_df.to_csv(evaluation_dir / "test_metrics.csv", index=False)
        horizon_metrics_df.to_csv(evaluation_dir / "horizon_mae.csv", index=False)
        prediction_df.to_csv(evaluation_dir / "test_predictions_original_scale.csv", index=False)
        plot_metrics_comparison(metrics_df, figure_dir / "test_metrics_comparison.png")
        plot_horizon_mae(horizon_metrics_df, artifacts["metadata"]["pred_len"], figure_dir / "horizon_mae.png")
        plot_actual_vs_prediction_horizon_1(
            y_test,
            predictions,
            artifacts["test"]["target_start"],
            figure_dir / "actual_vs_prediction_horizon_1.png",
        )
        plot_actual_vs_predicted_scatter(y_test, predictions, figure_dir / "actual_vs_predicted_scatter.png")
        plt.close("all")

    return {
        "metrics": metrics_df,
        "horizon_metrics": horizon_metrics_df,
        "predictions": prediction_df,
        "predictions_arrays": predictions,
        "predictions_scaled": predictions_scaled,
        "evaluation_dir": evaluation_dir,
        "figure_dir": figure_dir,
    }
