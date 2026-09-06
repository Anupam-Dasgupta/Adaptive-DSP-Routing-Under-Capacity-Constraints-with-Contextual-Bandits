"""Avazu loading and bounded, leakage-safe feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

TARGET_COLUMN = "click"
TIME_COLUMN = "datetime"
SOURCE_INDEX_COLUMN = "__index_level_0__"

CATEGORICAL_COLUMNS = (
    "C1",
    "banner_pos",
    "site_category",
    "app_category",
    "device_type",
    "device_conn_type",
    "C18",
)
NUMERIC_FEATURE_COLUMNS = (
    "intercept",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "log_ad_width",
    "log_ad_height",
    "aspect_ratio",
)
RAW_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + ("C15", "C16", TIME_COLUMN)
DEFAULT_LOAD_COLUMNS = tuple(
    dict.fromkeys(
        RAW_FEATURE_COLUMNS
        + (TARGET_COLUMN, "hour", "date", SOURCE_INDEX_COLUMN, "id")
    )
)
LEAKAGE_COLUMNS = {
    TARGET_COLUMN,
    "id",
    "hour",
    TIME_COLUMN,
    "date",
    SOURCE_INDEX_COLUMN,
}


def _source_paths(data_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(data_dir)
    paths = (
        directory / "avazu_train.parquet",
        directory / "avazu_test.parquet",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        expected = "\n".join(f"  - {path}" for path in paths)
        raise FileNotFoundError(
            "Avazu Parquet files were not found. Expected:\n" + expected
        )
    return paths


def _validate_parquet_schema(paths: Iterable[Path], required: set[str]) -> None:
    schemas = []
    for path in paths:
        names = tuple(pq.ParquetFile(path).schema_arrow.names)
        schemas.append(names)
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    if len(set(schemas)) != 1:
        raise ValueError("The Avazu source partitions do not have identical schemas.")


def sort_chronologically(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable chronological copy, using the source index to break ties."""

    if TIME_COLUMN not in frame:
        raise ValueError(f"Missing chronological column: {TIME_COLUMN}")
    result = frame.copy()
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], errors="raise")
    sort_columns = [TIME_COLUMN]
    if SOURCE_INDEX_COLUMN in result:
        if result[SOURCE_INDEX_COLUMN].duplicated().any():
            raise ValueError(f"{SOURCE_INDEX_COLUMN} must be unique.")
        sort_columns.append(SOURCE_INDEX_COLUMN)
    return result.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def load_avazu(
    data_dir: str | Path = "data/raw",
    *,
    n_events: int | None = 500_000,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the source partitions as one chronological Avazu event stream.

    The files called ``train`` and ``test`` are source partitions only. Both cover
    the same dates, so they are concatenated before the stable chronological sort.
    ``n_events`` then takes the earliest prefix, never a random sample.
    """

    if n_events is not None and n_events <= 0:
        raise ValueError("n_events must be positive or None.")

    requested = tuple(columns) if columns is not None else DEFAULT_LOAD_COLUMNS
    internal = tuple(dict.fromkeys(requested + (TIME_COLUMN, SOURCE_INDEX_COLUMN)))
    required = set(internal).union({"hour"})
    paths = _source_paths(data_dir)
    _validate_parquet_schema(paths, required)

    read_columns = tuple(dict.fromkeys(internal + ("hour",)))
    frames = [pd.read_parquet(path, columns=list(read_columns)) for path in paths]
    frame = pd.concat(frames, ignore_index=True, copy=False)

    if frame[SOURCE_INDEX_COLUMN].duplicated().any():
        raise ValueError("Source partitions overlap: duplicate source indices found.")
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="raise")
    encoded_hour = frame[TIME_COLUMN].dt.strftime("%y%m%d%H").astype("int64")
    if not encoded_hour.equals(frame["hour"].astype("int64")):
        raise ValueError("The integer hour column disagrees with datetime.")

    frame = sort_chronologically(frame)
    if n_events is not None:
        frame = frame.iloc[:n_events].copy()
    return frame.loc[:, list(requested)].reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an ordered stream into a training prefix and later evaluation suffix."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    ordered = sort_chronologically(frame)
    boundary = int(len(ordered) * train_fraction)
    if boundary == 0 or boundary == len(ordered):
        raise ValueError("The split would create an empty partition.")
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def chronological_sample(frame: pd.DataFrame, n_events: int) -> pd.DataFrame:
    """Systematically thin an ordered stream while retaining its full time span."""

    if n_events <= 0:
        raise ValueError("n_events must be positive.")
    ordered = sort_chronologically(frame)
    if n_events >= len(ordered):
        return ordered
    positions = np.linspace(0, len(ordered) - 1, n_events, dtype="int64")
    return ordered.iloc[positions].reset_index(drop=True)


def make_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the small model-facing table without fitting category statistics."""

    missing = sorted(set(RAW_FEATURE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing raw feature columns: {missing}")

    timestamp = pd.to_datetime(frame[TIME_COLUMN], errors="raise")
    hour_angle = 2.0 * np.pi * timestamp.dt.hour.to_numpy() / 24.0
    weekday_angle = 2.0 * np.pi * timestamp.dt.dayofweek.to_numpy() / 7.0
    width = frame["C15"].astype("float64").to_numpy()
    height = frame["C16"].astype("float64").to_numpy()

    result = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL_COLUMNS:
        result[column] = frame[column].astype("string").fillna("<MISSING>")
    result["intercept"] = 1.0
    result["hour_sin"] = np.sin(hour_angle)
    result["hour_cos"] = np.cos(hour_angle)
    result["weekday_sin"] = np.sin(weekday_angle)
    result["weekday_cos"] = np.cos(weekday_angle)
    result["log_ad_width"] = np.log1p(np.clip(width, 0.0, None))
    result["log_ad_height"] = np.log1p(np.clip(height, 0.0, None))
    result["aspect_ratio"] = np.divide(
        width,
        height,
        out=np.zeros_like(width, dtype="float64"),
        where=height != 0,
    )
    return result


class AvazuFeaturePipeline(BaseEstimator, TransformerMixin):
    """Fit bounded one-hot categories on a chronological training prefix only."""

    def __init__(self, max_categories: int = 8):
        self.max_categories = max_categories

    def fit(self, frame: pd.DataFrame, y: object = None) -> "AvazuFeaturePipeline":
        del y
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2.")
        features = make_feature_frame(frame)
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=self.max_categories,
            sparse_output=True,
            dtype=np.float64,
        )
        self.transformer_ = ColumnTransformer(
            transformers=[
                ("categorical", encoder, list(CATEGORICAL_COLUMNS)),
                ("numeric", "passthrough", list(NUMERIC_FEATURE_COLUMNS)),
            ],
            remainder="drop",
            sparse_threshold=1.0,
            verbose_feature_names_out=False,
        )
        self.transformer_.fit(features)
        self.n_features_out_ = len(self.transformer_.get_feature_names_out())
        if self.n_features_out_ > 64:
            raise RuntimeError(
                f"Feature dimension {self.n_features_out_} exceeds the 64-column limit."
            )
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        check_is_fitted(self, "transformer_")
        matrix = self.transformer_.transform(make_feature_frame(frame))
        return sparse.csr_matrix(matrix, dtype=np.float64)

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, "transformer_")
        names = self.transformer_.get_feature_names_out()
        leaked = LEAKAGE_COLUMNS.intersection(names)
        if leaked:
            raise RuntimeError(f"Leakage columns entered the feature matrix: {leaked}")
        return names
