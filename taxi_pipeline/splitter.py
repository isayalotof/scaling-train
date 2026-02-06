from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from taxi_pipeline.config import Config

DRIFT_THRESHOLD_PCT = 20.0


class TimeSplitter:
    def __init__(self, config: Config) -> None:
        self._config = config

    def split_holdout(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        df = df.sort(self._config.datetime_column)
        n = df.height
        split_idx = int(n * (1 - self._config.holdout_fraction))

        train_df = df.slice(0, split_idx)
        holdout_df = df.slice(split_idx, n - split_idx)
        return train_df, holdout_df

    def get_cv_splits(self, n_samples: int) -> TimeSeriesSplit:
        return TimeSeriesSplit(n_splits=self._config.n_cv_splits)

    @staticmethod
    def check_target_drift(
        y_train: np.ndarray, y_test: np.ndarray, fold: int
    ) -> None:
        mean_train = float(np.mean(y_train))
        mean_test = float(np.mean(y_test))
        if mean_train == 0:
            return
        drift_pct = abs(mean_train - mean_test) / abs(mean_train) * 100
        if drift_pct > DRIFT_THRESHOLD_PCT:
            warnings.warn(
                f"Fold {fold}: target drift {drift_pct:.1f}% exceeds {DRIFT_THRESHOLD_PCT}% threshold",
                stacklevel=2,
            )
