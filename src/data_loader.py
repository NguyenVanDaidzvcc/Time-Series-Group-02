from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_FILE = "Metro_Interstate_Traffic_Volume.csv"


def find_first_existing_path(paths: Iterable[Path]) -> Path:
    """Return the first existing path from a list of candidates."""
    for path in paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("Khong tim thay file trong cac duong dan du kien.")


def resolve_raw_data_path(
    data_file: str = DATA_FILE,
    cwd: Path | None = None,
    extra_candidates: Iterable[Path] | None = None,
) -> Path:
    """Resolve the raw Metro traffic CSV path from notebook or project root."""
    base = Path.cwd() if cwd is None else Path(cwd)
    candidates = [
        base / ".." / "data" / "raw" / data_file,
        base / "data" / "raw" / data_file,
        base / "time_series _" / "data" / "raw" / data_file,
    ]
    if extra_candidates is not None:
        candidates.extend(extra_candidates)
    return find_first_existing_path(candidates)


def project_dir_from_raw_path(raw_path: Path) -> Path:
    """Return project directory from .../data/raw/file.csv."""
    return Path(raw_path).resolve().parents[2]


def load_raw_traffic_data(path: Path | None = None, parse_dates: bool = True) -> pd.DataFrame:
    """Load the raw traffic dataset."""
    raw_path = resolve_raw_data_path() if path is None else Path(path)
    parse_cols = ["date_time"] if parse_dates else None
    return pd.read_csv(raw_path, parse_dates=parse_cols)


def clean_datetime(
    df: pd.DataFrame,
    datetime_col: str = "date_time",
    sort: bool = True,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Parse datetime column, optionally drop invalid rows and sort by time."""
    result = df.copy()
    result[datetime_col] = pd.to_datetime(result[datetime_col], errors="coerce")
    if drop_invalid:
        result = result.dropna(subset=[datetime_col]).copy()
    if sort:
        result = result.sort_values(datetime_col).reset_index(drop=True)
    return result


def get_season(month: int) -> str:
    """Map month number to meteorological season."""
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def add_eda_time_columns(df: pd.DataFrame, datetime_col: str = "date_time") -> pd.DataFrame:
    """Add time columns used in the exploration notebook."""
    result = df.copy()
    dt = result[datetime_col]
    result["hour"] = dt.dt.hour
    result["day_of_week"] = dt.dt.dayofweek
    result["day_name"] = dt.dt.day_name()
    result["month"] = dt.dt.month
    result["month_name"] = dt.dt.month_name()
    result["year"] = dt.dt.year
    result["date"] = dt.dt.date
    result["is_weekend"] = result["day_of_week"].isin([5, 6])
    result["season"] = result["month"].map(get_season)
    return result


def date_time_summary(df: pd.DataFrame, datetime_col: str = "date_time") -> pd.DataFrame:
    """Summarize date range, unique timestamps and duplicated timestamps."""
    return pd.DataFrame(
        {
            "min": [df[datetime_col].min()],
            "max": [df[datetime_col].max()],
            "so_moc_thoi_gian": [df[datetime_col].nunique()],
            "so_dong_trung_date_time": [df.duplicated(datetime_col).sum()],
            "da_sap_xep_tang_dan": [df[datetime_col].is_monotonic_increasing],
        }
    )


def missing_timestamps(df: pd.DataFrame, datetime_col: str = "date_time", freq: str = "h") -> pd.DatetimeIndex:
    """Return expected hourly timestamps missing from the data."""
    ordered = df.sort_values(datetime_col)
    expected_index = pd.date_range(ordered[datetime_col].min(), ordered[datetime_col].max(), freq=freq)
    return expected_index.difference(ordered[datetime_col])


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by column."""
    return pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": df.isna().mean() * 100,
        }
    ).sort_values("missing_count", ascending=False)


def duplicate_summary(df: pd.DataFrame, datetime_col: str = "date_time") -> dict[str, int]:
    """Return duplicate counts for the whole row and timestamp column."""
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_date_time": int(df.duplicated(subset=[datetime_col]).sum()),
    }


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute IQR outlier summary for selected numeric columns."""
    rows = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(mask.sum()),
                "outlier_percent": mask.mean() * 100,
                "min": df[col].min(),
                "max": df[col].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_percent", ascending=False)


def traffic_extreme_masks(
    df: pd.DataFrame,
    target_col: str = "traffic_volume",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, pd.Series]:
    """Return masks for IQR outliers, percentile extremes and zero traffic."""
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr
    p_low = df[target_col].quantile(lower_quantile)
    p_high = df[target_col].quantile(upper_quantile)
    return {
        "iqr_outlier": (df[target_col] < iqr_lower) | (df[target_col] > iqr_upper),
        "percentile_extreme": (df[target_col] <= p_low) | (df[target_col] >= p_high),
        "zero_traffic": df[target_col] == 0,
    }


def time_group_summary(
    df: pd.DataFrame,
    target_col: str = "traffic_volume",
    groups: tuple[str, ...] = ("hour", "day_name", "month", "year"),
) -> pd.DataFrame:
    """Aggregate target statistics by common time groups."""
    pieces = []
    keys = []
    for group in groups:
        if group not in df.columns:
            continue
        pieces.append(df.groupby(group)[target_col].agg(["count", "mean", "median", "std"]).assign(group=group))
        keys.append(group)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, keys=keys)


def numeric_correlation(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str = "traffic_volume",
) -> pd.DataFrame:
    """Return correlations sorted by association with target."""
    corr_df = df[columns].copy()
    if "is_weekend" in corr_df.columns:
        corr_df["is_weekend"] = corr_df["is_weekend"].astype(int)
    corr_matrix = corr_df.corr(numeric_only=True)
    return corr_matrix[target_col].sort_values(ascending=False).to_frame(f"corr_with_{target_col}")
