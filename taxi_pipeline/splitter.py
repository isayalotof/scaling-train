"""Chronological data splitting for time-series validation."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from taxi_pipeline.config import Config

logger = logging.getLogger("TaxiPipeline")


class TimeSplitter:
    """Chronological data splitting to prevent data leakage.

    Sorts data by pickup datetime, carves out a holdout set from the tail,
    and provides TimeSeriesSplit folds on the remaining data.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def split_holdout(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Reserve the latest fraction of data as holdout.

        The holdout is sliced **before** any CV or training.

        Args:
            df: Full feature-engineered DataFrame sorted by time.

        Returns:
            (train_df, holdout_df) tuple.
        """
        df = df.sort(self._config.datetime_column)
        n = df.height
        split_idx = int(n * (1 - self._config.holdout_fraction))

        train_df = df.slice(0, split_idx)
        holdout_df = df.slice(split_idx, n - split_idx)

        train_min = train_df[self._config.datetime_column].min()
        train_max = train_df[self._config.datetime_column].max()
        hold_min = holdout_df[self._config.datetime_column].min()
        hold_max = holdout_df[self._config.datetime_column].max()

        logger.info(
            "Train period: %s to %s (%d rows)",
            train_min, train_max, train_df.height,
        )
        logger.info(
            "Holdout period: %s to %s (%d rows)",
            hold_min, hold_max, holdout_df.height,
        )

        return train_df, holdout_df

    def get_cv_splits(self, n_samples: int) -> TimeSeriesSplit:
        """Return a scikit-learn TimeSeriesSplit object.

        Args:
            n_samples: Number of samples in the training set.

        Returns:
            Configured TimeSeriesSplit instance.
        """
        return TimeSeriesSplit(n_splits=self._config.n_cv_splits)

    @staticmethod
    def check_target_drift(
        y_train: np.ndarray, y_test: np.ndarray, fold: int
    ) -> None:
        """Log a warning if train/test target means differ by more than 20%.

        Args:
            y_train: Target values in training fold.
            y_test: Target values in validation fold.
            fold: Current fold number (for logging).
        """
        mean_train = float(np.mean(y_train))
        mean_test = float(np.mean(y_test))
        if mean_train == 0:
            return
        drift_pct = abs(mean_train - mean_test) / abs(mean_train) * 100
        logger.info(
            "Fold %d target mean: train=%.4f, test=%.4f (drift=%.1f%%)",
            fold, mean_train, mean_test, drift_pct,
        )
        if drift_pct > 20:
            logger.warning(
                "Fold %d: target drift %.1f%% exceeds 20%% threshold!",
                fold, drift_pct,
            )
