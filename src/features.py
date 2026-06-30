from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import clean_datetime, project_dir_from_raw_path, resolve_raw_data_path
except ImportError:
    from data_loader import clean_datetime, project_dir_from_raw_path, resolve_raw_data_path


NUMERIC_AGG_COLUMNS = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
CATEGORICAL_COLUMNS = ["holiday", "weather_main", "weather_description"]
TARGET_COL = "traffic_volume"

BASE_FEATURE_COLS = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "is_holiday",
]


def mode_or_first(series: pd.Series):
    """Return mode value, or NaN when the series has no non-null values."""
    mode_values = series.dropna().mode()
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    return np.nan


def aggregate_duplicate_timestamps(df: pd.DataFrame, datetime_col: str = "date_time") -> pd.DataFrame:
    """Collapse duplicate timestamps using mean for numeric and mode for categorical columns."""
    numeric_agg = {col: "mean" for col in NUMERIC_AGG_COLUMNS}
    categorical_agg = {col: mode_or_first for col in CATEGORICAL_COLUMNS}
    return (
        df.groupby(datetime_col, as_index=False)
        .agg({**numeric_agg, **categorical_agg})
        .sort_values(datetime_col)
        .reset_index(drop=True)
    )


def resample_hourly(df: pd.DataFrame, datetime_col: str = "date_time", freq: str = "h") -> pd.DataFrame:
    """Create a continuous hourly dataframe."""
    return df.set_index(datetime_col).asfreq(freq).reset_index()


def fill_missing_hourly_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric values by interpolation and categoricals by forward/backward fill."""
    result = df.copy()
    result[NUMERIC_AGG_COLUMNS] = result[NUMERIC_AGG_COLUMNS].interpolate(method="linear", limit_direction="both")
    result[CATEGORICAL_COLUMNS] = result[CATEGORICAL_COLUMNS].ffill().bfill().fillna("Unknown")
    result["holiday"] = result["holiday"].fillna("None")
    return result


def add_calendar_features(df: pd.DataFrame, datetime_col: str = "date_time") -> pd.DataFrame:
    """Add calendar features used by the forecasting models."""
    result = df.copy()
    dt = result[datetime_col]
    result["hour"] = dt.dt.hour
    result["day_of_week"] = dt.dt.dayofweek
    result["day_of_month"] = dt.dt.day
    result["day_of_year"] = dt.dt.dayofyear
    result["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    result["month"] = dt.dt.month
    result["quarter"] = dt.dt.quarter
    result["year"] = dt.dt.year
    return result


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for periodic time columns."""
    result = df.copy()
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_of_week_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["day_of_year_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["day_of_year_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    return result


def add_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekend and holiday indicator features."""
    result = df.copy()
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["is_holiday"] = (result["holiday"].fillna("None") != "None").astype(int)
    return result


def add_weather_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode weather_main and return the created column names."""
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather_main", dtype=int)
    result = pd.concat([df.copy(), weather_dummies], axis=1)
    return result, weather_dummies.columns.tolist()


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the model-ready feature dataframe from raw rows."""
    prepared = clean_datetime(df)
    hourly = aggregate_duplicate_timestamps(prepared)
    hourly = resample_hourly(hourly)
    hourly = fill_missing_hourly_values(hourly)
    features = add_calendar_features(hourly)
    features = add_cyclical_features(features)
    features = add_binary_features(features)
    features, weather_feature_cols = add_weather_dummies(features)
    feature_cols = BASE_FEATURE_COLS + weather_feature_cols
    model_df = features[["date_time"] + feature_cols + [TARGET_COL]].copy()
    model_df[feature_cols + [TARGET_COL]] = model_df[feature_cols + [TARGET_COL]].astype(float)
    return model_df, feature_cols, weather_feature_cols


def chronological_split(
    model_df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series dataframe chronologically into train/validation/test."""
    n_rows = len(model_df)
    train_end = int(n_rows * train_ratio)
    validation_end = int(n_rows * (train_ratio + validation_ratio))
    return (
        model_df.iloc[:train_end].copy(),
        model_df.iloc[train_end:validation_end].copy(),
        model_df.iloc[validation_end:].copy(),
    )


def scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
) -> dict[str, object]:
    """Fit scalers on train split and transform train/validation/test splits."""
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    return {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "X_train": feature_scaler.fit_transform(train_df[feature_cols]),
        "X_val": feature_scaler.transform(val_df[feature_cols]),
        "X_test": feature_scaler.transform(test_df[feature_cols]),
        "y_train": target_scaler.fit_transform(train_df[[target_col]]).ravel(),
        "y_val": target_scaler.transform(val_df[[target_col]]).ravel(),
        "y_test": target_scaler.transform(test_df[[target_col]]).ravel(),
    }


def create_windows(
    X: np.ndarray,
    y: np.ndarray,
    date_time_values: np.ndarray,
    seq_len: int = 168,
    pred_len: int = 24,
) -> dict[str, np.ndarray]:
    """Create sliding windows for multi-step forecasting."""
    X_windows = []
    y_windows = []
    start_times = []
    target_start_times = []
    target_end_times = []
    max_start = len(X) - seq_len - pred_len + 1

    for start in range(max(0, max_start)):
        input_end = start + seq_len
        target_end = input_end + pred_len
        X_windows.append(X[start:input_end])
        y_windows.append(y[input_end:target_end])
        start_times.append(date_time_values[start])
        target_start_times.append(date_time_values[input_end])
        target_end_times.append(date_time_values[target_end - 1])

    return {
        "X": np.asarray(X_windows, dtype=np.float32),
        "y": np.asarray(y_windows, dtype=np.float32),
        "window_start": np.asarray(start_times, dtype="datetime64[ns]"),
        "target_start": np.asarray(target_start_times, dtype="datetime64[ns]"),
        "target_end": np.asarray(target_end_times, dtype="datetime64[ns]"),
    }


def build_windows_for_splits(
    scaled: dict[str, object],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int = 168,
    pred_len: int = 24,
) -> dict[str, dict[str, np.ndarray]]:
    """Create train/validation/test windows from scaled arrays."""
    return {
        "train": create_windows(scaled["X_train"], scaled["y_train"], train_df["date_time"].to_numpy(), seq_len, pred_len),
        "validation": create_windows(scaled["X_val"], scaled["y_val"], val_df["date_time"].to_numpy(), seq_len, pred_len),
        "test": create_windows(scaled["X_test"], scaled["y_test"], test_df["date_time"].to_numpy(), seq_len, pred_len),
    }


def save_feature_artifacts(
    processed_dir: Path,
    raw_path: Path,
    model_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaled: dict[str, object],
    windows: dict[str, dict[str, np.ndarray]],
    feature_cols: list[str],
    weather_feature_cols: list[str],
    seq_len: int,
    pred_len: int,
    target_col: str = TARGET_COL,
) -> dict:
    """Save processed datasets, windows, scalers and feature metadata."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(processed_dir / "traffic_features_hourly.csv", index=False)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "validation.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    np.savez_compressed(processed_dir / "train_windows.npz", **windows["train"])
    np.savez_compressed(processed_dir / "validation_windows.npz", **windows["validation"])
    np.savez_compressed(processed_dir / "test_windows.npz", **windows["test"])

    joblib.dump(scaled["feature_scaler"], processed_dir / "feature_scaler.pkl")
    joblib.dump(scaled["target_scaler"], processed_dir / "target_scaler.pkl")

    metadata = {
        "raw_path": str(raw_path),
        "processed_dir": str(processed_dir),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "weather_feature_cols": weather_feature_cols,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "window_shapes": {
            "train_X": list(windows["train"]["X"].shape),
            "train_y": list(windows["train"]["y"].shape),
            "validation_X": list(windows["validation"]["X"].shape),
            "validation_y": list(windows["validation"]["y"].shape),
            "test_X": list(windows["test"]["X"].shape),
            "test_y": list(windows["test"]["y"].shape),
        },
    }

    with open(processed_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(processed_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    return metadata


def run_feature_pipeline(
    raw_path: Path | None = None,
    processed_dir: Path | None = None,
    seq_len: int = 168,
    pred_len: int = 24,
) -> dict[str, object]:
    """Run the full feature engineering pipeline and save artifacts."""
    resolved_raw_path = resolve_raw_data_path() if raw_path is None else Path(raw_path).resolve()
    project_dir = project_dir_from_raw_path(resolved_raw_path)
    resolved_processed_dir = project_dir / "data" / "processed" if processed_dir is None else Path(processed_dir)

    raw_df = pd.read_csv(resolved_raw_path)
    model_df, feature_cols, weather_feature_cols = build_feature_frame(raw_df)
    train_df, val_df, test_df = chronological_split(model_df)
    scaled = scale_splits(train_df, val_df, test_df, feature_cols)
    windows = build_windows_for_splits(scaled, train_df, val_df, test_df, seq_len, pred_len)
    metadata = save_feature_artifacts(
        resolved_processed_dir,
        resolved_raw_path,
        model_df,
        train_df,
        val_df,
        test_df,
        scaled,
        windows,
        feature_cols,
        weather_feature_cols,
        seq_len,
        pred_len,
    )
    return {
        "metadata": metadata,
        "model_df": model_df,
        "splits": {"train": train_df, "validation": val_df, "test": test_df},
        "windows": windows,
        "processed_dir": resolved_processed_dir,
    }
