"""Production-grade ML pipeline for NYC Taxi fare prediction.

Modular pipeline with time-series validation, Optuna hyperparameter tuning,
and LightGBM training. Designed to prevent data leakage through chronological
splitting and optimize resources via two-stage training.
"""

from __future__ import annotations

import logging
import os
import pickle
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TaxiPipeline")


# ---------------------------------------------------------------------------
# 3.1  Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Central configuration for the entire pipeline.

    Args:
        data_paths: List of parquet file paths to load.
        datetime_column: Column used for temporal ordering.
        target_column: Prediction target.
        n_cv_splits: Number of folds for TimeSeriesSplit.
        n_trials: Number of Optuna trials.
        tuning_sample_fraction: Fraction of sorted data used for tuning (Stage 1).
        holdout_fraction: Fraction of latest data reserved for final evaluation.
        high_value_threshold: Fare above this value gets flagged, not dropped.
        max_passengers: Upper passenger count limit for hard drop.
        min_fare: Minimum fare for hard drop.
        max_fare_hard: Maximum fare for hard drop.
        max_distance: Maximum trip distance for hard drop.
        feature_columns: Columns used as model features.
        categorical_features: Columns treated as categorical by LightGBM.
        model_save_path: Where to persist the trained model.
        config_save_path: Where to persist the config snapshot.
        random_seed: Reproducibility seed.
    """

    data_paths: list[str] = field(default_factory=list)
    datetime_column: str = "tpep_pickup_datetime"
    dropoff_column: str = "tpep_dropoff_datetime"
    target_column: str = "fare_amount"

    # Time split params
    n_cv_splits: int = 5

    # Optimization params
    n_trials: int = 30
    tuning_sample_fraction: float = 0.2

    # Holdout
    holdout_fraction: float = 0.15

    # Cleaning thresholds
    high_value_threshold: float = 500.0
    max_passengers: int = 10
    min_fare: float = 0.0
    max_fare_hard: float = 5000.0
    max_distance: float = 500.0

    # Features
    feature_columns: list[str] = field(default_factory=lambda: [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "pickup_hour",
        "pickup_day",
        "pickup_month",
        "pickup_weekday",
        "trip_duration_seconds",
        "is_rush_hour",
        "is_weekend",
        "is_night",
        "is_high_value_trip",
    ])

    categorical_features: list[str] = field(default_factory=lambda: [
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "pickup_weekday",
    ])

    # Persistence
    model_save_path: str = "model.pkl"
    config_save_path: str = "config.json"

    random_seed: int = 42


# ---------------------------------------------------------------------------
# 3.2  Data Processor
# ---------------------------------------------------------------------------
class DataProcessor:
    """Handles loading, type optimization, cleaning, and feature engineering.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def load_and_optimize(self) -> pl.DataFrame:
        """Load parquet files and downcast types to minimise RAM.

        Returns:
            Concatenated and type-optimized DataFrame.
        """
        frames: list[pl.DataFrame] = []
        for path in self._config.data_paths:
            df = pl.read_parquet(path)
            df = df.rename({col: col.lower() for col in df.columns})
            frames.append(df)

        df = pl.concat(frames, how="vertical_relaxed")
        # Normalise column names back to spec casing for convenience
        rename_map = {
            "pulocationid": "PULocationID",
            "dolocationid": "DOLocationID",
        }
        df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

        mem_before = df.estimated_size("mb")

        df = df.with_columns([
            pl.col("passenger_count").cast(pl.Int8),
            pl.col("PULocationID").cast(pl.Int16),
            pl.col("DOLocationID").cast(pl.Int16),
            pl.col("payment_type").cast(pl.Int8),
            pl.col("trip_distance").cast(pl.Float32),
            pl.col("fare_amount").cast(pl.Float32),
        ])

        # Downcast remaining Float64 -> Float32
        float64_cols = [
            c for c in df.columns
            if df[c].dtype == pl.Float64
        ]
        if float64_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.Float32) for c in float64_cols
            ])

        # Downcast remaining Int64 -> Int32
        int64_cols = [
            c for c in df.columns
            if df[c].dtype == pl.Int64
        ]
        if int64_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.Int32) for c in int64_cols
            ])

        mem_after = df.estimated_size("mb")
        logger.info(
            "Memory: %.2f MB -> %.2f MB (saved %.1f%%)",
            mem_before,
            mem_after,
            (1 - mem_after / mem_before) * 100,
        )
        return df

    def sanity_check_and_clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply hard-drop filters and soft-flag high-value trips.

        Hard drops: negative fare, zero distance, passenger_count 0 or > max.
        Soft flag: fare > threshold -> ``is_high_value_trip = True``.

        Args:
            df: Raw (type-optimized) DataFrame.

        Returns:
            Cleaned DataFrame with ``is_high_value_trip`` column.
        """
        rows_before = df.height

        # Drop nulls in critical columns
        critical_cols = [
            self._config.target_column,
            "trip_distance",
            self._config.datetime_column,
            self._config.dropoff_column,
        ]
        df = df.drop_nulls(subset=[c for c in critical_cols if c in df.columns])

        # Hard drops
        df = df.filter(
            (pl.col(self._config.target_column) > self._config.min_fare)
            & (pl.col("trip_distance") > 0)
            & (pl.col("passenger_count") > 0)
            & (pl.col("passenger_count") <= self._config.max_passengers)
            & (pl.col(self._config.target_column) < self._config.max_fare_hard)
            & (pl.col("trip_distance") < self._config.max_distance)
        )

        rows_after = df.height
        logger.info(
            "Cleaning: %d -> %d rows (dropped %d, %.2f%%)",
            rows_before,
            rows_after,
            rows_before - rows_after,
            (rows_before - rows_after) / rows_before * 100,
        )

        # Soft flag
        df = df.with_columns(
            (pl.col(self._config.target_column) > self._config.high_value_threshold)
            .cast(pl.Int8)
            .alias("is_high_value_trip")
        )
        high_count = df.filter(pl.col("is_high_value_trip") == 1).height
        logger.info("High-value trips flagged: %d", high_count)

        return df

    def feature_engineering(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create temporal and derived features.

        Features created: pickup_hour, pickup_day, pickup_month, pickup_weekday,
        trip_duration_seconds, is_rush_hour, is_weekend, is_night.

        Args:
            df: Cleaned DataFrame.

        Returns:
            DataFrame with engineered features.
        """
        df = df.with_columns([
            pl.col(self._config.datetime_column).dt.hour().cast(pl.Int8).alias("pickup_hour"),
            pl.col(self._config.datetime_column).dt.day().cast(pl.Int8).alias("pickup_day"),
            pl.col(self._config.datetime_column).dt.month().cast(pl.Int8).alias("pickup_month"),
            pl.col(self._config.datetime_column).dt.weekday().cast(pl.Int8).alias("pickup_weekday"),
            (
                (pl.col(self._config.dropoff_column) - pl.col(self._config.datetime_column))
                .dt.total_seconds()
            ).cast(pl.Float32).alias("trip_duration_seconds"),
        ])

        df = df.with_columns([
            (
                ((pl.col("pickup_hour") >= 7) & (pl.col("pickup_hour") <= 9))
                | ((pl.col("pickup_hour") >= 17) & (pl.col("pickup_hour") <= 19))
            ).cast(pl.Int8).alias("is_rush_hour"),
            (pl.col("pickup_weekday") >= 6).cast(pl.Int8).alias("is_weekend"),
            (
                (pl.col("pickup_hour") >= 23) | (pl.col("pickup_hour") <= 5)
            ).cast(pl.Int8).alias("is_night"),
        ])

        # Filter out non-positive durations (data errors)
        df = df.filter(pl.col("trip_duration_seconds") > 0)

        logger.info("Feature engineering complete. Columns: %d, Rows: %d", len(df.columns), df.height)
        return df


# ---------------------------------------------------------------------------
# 3.3  Time Splitter
# ---------------------------------------------------------------------------
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

    def get_cv_splits(
        self, n_samples: int
    ) -> TimeSeriesSplit:
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


# ---------------------------------------------------------------------------
# 3.4  Model Trainer
# ---------------------------------------------------------------------------
class ModelTrainer:
    """Two-stage LightGBM trainer with Optuna hyperparameter search.

    Stage 1: tune on a subsample with pruning.
    Stage 2: retrain on full training data with best params.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    # -- helpers -------------------------------------------------------------

    def _prepare_arrays(
        self, df: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract feature matrix X and target vector y as numpy arrays.

        Args:
            df: DataFrame with feature and target columns.

        Returns:
            (X, y, feature_names) tuple.
        """
        available_features = [
            c for c in self._config.feature_columns if c in df.columns
        ]
        X = df.select(available_features).to_pandas()
        y = df.select(self._config.target_column).to_numpy().ravel()
        return X, y, available_features

    def _detect_gpu(self) -> str:
        """Return 'gpu' if a CUDA device is available, else 'cpu'."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info("GPU detected: %s", result.stdout.strip().split("\n")[0])
                return "gpu"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        logger.info("No GPU detected, using CPU")
        return "cpu"

    # -- Stage 1 -------------------------------------------------------------

    def tune_hyperparameters(
        self, train_df: pl.DataFrame, splitter: TimeSplitter
    ) -> dict[str, Any]:
        """Stage 1: Optuna search on a data subsample.

        Args:
            train_df: Training DataFrame (sorted by time, excluding holdout).
            splitter: TimeSplitter instance for CV strategy.

        Returns:
            Best hyperparameter dict found by Optuna.
        """
        sample_size = int(train_df.height * self._config.tuning_sample_fraction)
        # Take the first N rows (chronologically earliest) to keep temporal order
        sample_df = train_df.head(sample_size)
        logger.info(
            "Stage 1: tuning on %d rows (%.0f%% of train)",
            sample_df.height,
            self._config.tuning_sample_fraction * 100,
        )

        X_sample, y_sample, feature_names = self._prepare_arrays(sample_df)
        tscv = splitter.get_cv_splits(X_sample.shape[0])
        device = self._detect_gpu()

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "objective": "regression",
                "metric": "rmse",
                "device": device,
                "verbosity": -1,
                "random_state": self._config.random_seed,
                "n_estimators": 1000,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            rmse_scores: list[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_sample)):
                X_tr, X_val = X_sample.iloc[train_idx], X_sample.iloc[val_idx]
                y_tr, y_val = y_sample[train_idx], y_sample[val_idx]

                splitter.check_target_drift(y_tr, y_val, fold_idx)

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        LightGBMPruningCallback(trial, "rmse"),
                    ],
                )

                y_pred = model.predict(X_val)
                fold_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
                rmse_scores.append(fold_rmse)
                logger.info(
                    "Trial %d, Fold %d: RMSE=%.4f",
                    trial.number, fold_idx, fold_rmse,
                )

            mean_rmse = float(np.mean(rmse_scores))
            return mean_rmse

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            study_name="taxi_fare_tuning",
        )
        study.optimize(objective, n_trials=self._config.n_trials, show_progress_bar=True)

        best_params = study.best_trial.params
        logger.info("Best trial RMSE: %.4f", study.best_value)
        logger.info("Best params: %s", best_params)
        return best_params

    # -- Stage 2 -------------------------------------------------------------

    def train_final_model(
        self,
        train_df: pl.DataFrame,
        best_params: dict[str, Any],
        splitter: TimeSplitter,
    ) -> lgb.LGBMRegressor:
        """Stage 2: Train on full training data with best hyperparameters.

        Uses the last CV fold as the early-stopping validation set.

        Args:
            train_df: Full training DataFrame (excluding holdout).
            best_params: Hyperparameters from Stage 1.
            splitter: TimeSplitter instance.

        Returns:
            Trained LGBMRegressor.
        """
        X_full, y_full, _ = self._prepare_arrays(train_df)
        device = self._detect_gpu()

        # Use last fold of TimeSeriesSplit for early stopping
        tscv = splitter.get_cv_splits(X_full.shape[0])
        train_idx, val_idx = list(tscv.split(X_full))[-1]
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]

        splitter.check_target_drift(y_tr, y_val, fold=-1)

        params: dict[str, Any] = {
            "objective": "regression",
            "metric": "rmse",
            "device": device,
            "verbosity": -1,
            "random_state": self._config.random_seed,
            "n_estimators": 2000,
            **best_params,
        }

        logger.info("Stage 2: training final model on %d rows, validating on %d", len(y_tr), len(y_val))
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        val_pred = model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        logger.info("Final model validation RMSE: %.4f", val_rmse)
        return model

    # -- Evaluation ----------------------------------------------------------

    def evaluate_holdout(
        self, model: lgb.LGBMRegressor, holdout_df: pl.DataFrame
    ) -> dict[str, float]:
        """Evaluate the final model on the time-separated holdout set.

        Args:
            model: Trained LGBMRegressor.
            holdout_df: Holdout DataFrame.

        Returns:
            Dictionary of metric name -> value.
        """
        X_hold, y_hold, _ = self._prepare_arrays(holdout_df)
        y_pred = model.predict(X_hold)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_hold, y_pred))),
            "mae": float(mean_absolute_error(y_hold, y_pred)),
            "r2": float(r2_score(y_hold, y_pred)),
        }
        logger.info(
            "Holdout metrics: RMSE=$%.4f, MAE=$%.4f, R2=%.4f",
            metrics["rmse"], metrics["mae"], metrics["r2"],
        )
        return metrics

    # -- Persistence ---------------------------------------------------------

    def save(
        self, model: lgb.LGBMRegressor, config: Config, metrics: dict[str, float]
    ) -> None:
        """Save model and config to disk.

        Args:
            model: Trained model.
            config: Pipeline config snapshot.
            metrics: Final holdout metrics.
        """
        with open(config.model_save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Model saved to %s", config.model_save_path)

        config_dict = asdict(config)
        config_dict["holdout_metrics"] = metrics
        with open(config.config_save_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info("Config saved to %s", config.config_save_path)


# ---------------------------------------------------------------------------
# 5.  Main Pipeline
# ---------------------------------------------------------------------------
def download_data(months: list[str], dest_dir: str = ".") -> list[str]:
    """Download NYC taxi parquet files if they do not already exist.

    Args:
        months: List of month strings like ['01', '02', ...].
        dest_dir: Directory to save files.

    Returns:
        List of local file paths.
    """
    import subprocess

    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-{}.parquet"
    paths: list[str] = []
    for month in months:
        filename = f"yellow_tripdata_2023-{month}.parquet"
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            url = base_url.format(month)
            logger.info("Downloading %s ...", filename)
            subprocess.run(["wget", "-q", url, "-O", filepath], check=True)
        else:
            logger.info("File exists: %s", filename)
        paths.append(filepath)
    return paths


def generate_synthetic_data(
    months: list[str],
    dest_dir: str = ".",
    rows_per_month: int = 100_000,
    seed: int = 42,
) -> list[str]:
    """Generate synthetic taxi data matching the real NYC schema.

    Used when real data is unavailable (e.g. no network access).
    Generates realistic distributions for fares, distances, and times.

    Args:
        months: List of month strings like ['01', '02', ...].
        dest_dir: Directory to write parquet files.
        rows_per_month: Number of rows per month file.
        seed: Random seed for reproducibility.

    Returns:
        List of generated file paths.
    """
    rng = np.random.default_rng(seed)
    paths: list[str] = []
    location_ids = list(range(1, 264))

    for month_str in months:
        month = int(month_str)
        year = 2023
        import calendar
        days_in_month = calendar.monthrange(year, month)[1]

        n = rows_per_month

        # Generate pickup timestamps spread across the month
        day_offsets = rng.integers(0, days_in_month, size=n)
        hour_probs = np.array([
            0.02, 0.015, 0.01, 0.008, 0.007, 0.01, 0.025, 0.05,
            0.06, 0.055, 0.05, 0.05, 0.055, 0.05, 0.055, 0.05,
            0.05, 0.06, 0.065, 0.055, 0.045, 0.04, 0.04, 0.03,
        ])
        hour_probs = hour_probs / hour_probs.sum()
        hour_offsets = rng.choice(24, size=n, p=hour_probs)
        minute_offsets = rng.integers(0, 60, size=n)
        second_offsets = rng.integers(0, 60, size=n)

        base = np.datetime64(f"{year}-{month:02d}-01", "ns")
        pickups = (
            base
            + day_offsets.astype("timedelta64[D]")
            + hour_offsets.astype("timedelta64[h]")
            + minute_offsets.astype("timedelta64[m]")
            + second_offsets.astype("timedelta64[s]")
        )

        # Trip distance: lognormal, mean ~3 miles
        trip_distance = rng.lognormal(mean=1.0, sigma=0.7, size=n).clip(0.1, 80)

        # Duration in seconds: correlated with distance
        base_speed_mph = rng.normal(12, 4, size=n).clip(3, 40)
        duration_hours = trip_distance / base_speed_mph
        duration_seconds = (duration_hours * 3600).astype(int).clip(60, 7200)

        dropoffs = pickups + duration_seconds.astype("timedelta64[s]")

        # Fare: base $3.50 + ~$2.50/mile + noise
        fare_amount = 3.50 + 2.50 * trip_distance + rng.normal(0, 1.5, size=n)
        fare_amount = fare_amount.clip(2.5, 300)

        # Inject some anomalies for cleaning tests
        anomaly_idx = rng.choice(n, size=int(n * 0.02), replace=False)
        half = len(anomaly_idx) // 2
        fare_amount[anomaly_idx[:half]] = rng.uniform(-50, -0.01, size=half)
        trip_distance_arr = trip_distance.copy()
        trip_distance_arr[anomaly_idx[half:]] = 0.0

        # Passenger count
        passenger_count = rng.choice(
            [0, 1, 2, 3, 4, 5, 6, 11],
            size=n,
            p=[0.01, 0.70, 0.12, 0.06, 0.04, 0.03, 0.03, 0.01],
        ).astype(np.float64)

        # Location IDs, payment type, vendor
        pu_loc = rng.choice(location_ids, size=n)
        do_loc = rng.choice(location_ids, size=n)
        payment_type = rng.choice([1, 2, 3, 4], size=n, p=[0.6, 0.3, 0.05, 0.05])
        vendor_id = rng.choice([1, 2], size=n)
        ratecode_id = rng.choice([1, 2, 3, 4, 5, 99], size=n, p=[0.9, 0.04, 0.02, 0.02, 0.01, 0.01]).astype(np.float64)

        # Surcharges
        extra = rng.choice([0.0, 0.5, 1.0, 2.5], size=n, p=[0.3, 0.3, 0.2, 0.2])
        mta_tax = np.full(n, 0.5)
        tip_upper = np.maximum(fare_amount * 0.3, 0.01)
        tip_amount = np.where(payment_type == 1, rng.uniform(0, tip_upper), 0.0)
        tolls_amount = np.where(rng.random(n) < 0.05, rng.uniform(5, 15, size=n), 0.0)
        improvement_surcharge = np.full(n, 1.0)
        congestion_surcharge = rng.choice([0.0, 2.5], size=n, p=[0.3, 0.7])
        airport_fee = rng.choice([0.0, 1.25], size=n, p=[0.85, 0.15])
        total_amount = fare_amount + extra + mta_tax + tip_amount + tolls_amount + improvement_surcharge + congestion_surcharge + airport_fee

        # Inject nulls in passenger_count / ratecode / congestion / airport
        null_idx = rng.choice(n, size=int(n * 0.023), replace=False)
        passenger_count_series = pl.Series("passenger_count", passenger_count).cast(pl.Float64)
        ratecode_series = pl.Series("RatecodeID", ratecode_id).cast(pl.Float64)
        congestion_series = pl.Series("congestion_surcharge", congestion_surcharge).cast(pl.Float64)
        airport_series = pl.Series("airport_fee", airport_fee).cast(pl.Float64)

        df = pl.DataFrame({
            "VendorID": vendor_id.astype(np.int64),
            "tpep_pickup_datetime": pickups,
            "tpep_dropoff_datetime": dropoffs,
            "passenger_count": passenger_count,
            "trip_distance": trip_distance_arr,
            "RatecodeID": ratecode_id,
            "store_and_fwd_flag": rng.choice(["Y", "N"], size=n, p=[0.01, 0.99]),
            "PULocationID": pu_loc.astype(np.int64),
            "DOLocationID": do_loc.astype(np.int64),
            "payment_type": payment_type.astype(np.int64),
            "fare_amount": fare_amount,
            "extra": extra,
            "mta_tax": mta_tax,
            "tip_amount": tip_amount,
            "tolls_amount": tolls_amount,
            "improvement_surcharge": improvement_surcharge,
            "total_amount": total_amount,
            "congestion_surcharge": congestion_surcharge,
            "airport_fee": airport_fee,
        })

        # Set some values to null
        mask = pl.Series("mask", np.isin(np.arange(n), null_idx))
        df = df.with_columns([
            pl.when(mask).then(None).otherwise(pl.col("passenger_count")).alias("passenger_count"),
            pl.when(mask).then(None).otherwise(pl.col("RatecodeID")).alias("RatecodeID"),
            pl.when(mask).then(None).otherwise(pl.col("congestion_surcharge")).alias("congestion_surcharge"),
            pl.when(mask).then(None).otherwise(pl.col("airport_fee")).alias("airport_fee"),
        ])

        df = df.sort("tpep_pickup_datetime")

        filepath = os.path.join(dest_dir, f"yellow_tripdata_2023-{month_str}.parquet")
        df.write_parquet(filepath)
        logger.info("Generated synthetic %s (%d rows)", filepath, n)
        paths.append(filepath)

    return paths


def resolve_data(months: list[str], dest_dir: str = ".") -> list[str]:
    """Get data file paths -- download if possible, otherwise generate synthetic.

    Args:
        months: List of month strings.
        dest_dir: Target directory.

    Returns:
        List of parquet file paths.
    """
    # Check if files already exist
    all_exist = all(
        os.path.exists(os.path.join(dest_dir, f"yellow_tripdata_2023-{m}.parquet"))
        for m in months
    )
    if all_exist:
        paths = [os.path.join(dest_dir, f"yellow_tripdata_2023-{m}.parquet") for m in months]
        logger.info("All %d data files found locally", len(paths))
        return paths

    # Try downloading
    try:
        return download_data(months, dest_dir)
    except Exception as exc:
        logger.warning("Download failed (%s), generating synthetic data", exc)
        return generate_synthetic_data(months, dest_dir)


def run_pipeline() -> None:
    """Orchestrate the full training pipeline."""
    # -- Download / locate data ---
    months = ["01", "02", "03"]
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_paths = resolve_data(months, dest_dir=data_dir)

    config = Config(data_paths=data_paths)

    # -- Data processing ---
    processor = DataProcessor(config)
    df = processor.load_and_optimize()
    df = processor.sanity_check_and_clean(df)
    df = processor.feature_engineering(df)

    # Sort by time
    df = df.sort(config.datetime_column)

    date_min = df[config.datetime_column].min()
    date_max = df[config.datetime_column].max()
    logger.info("Data range: from %s to %s", date_min, date_max)

    # -- Split holdout ---
    splitter = TimeSplitter(config)
    train_df, holdout_df = splitter.split_holdout(df)

    # -- Training ---
    trainer = ModelTrainer(config)

    logger.info("=" * 60)
    logger.info("STAGE 1: Hyperparameter tuning")
    logger.info("=" * 60)
    best_params = trainer.tune_hyperparameters(train_df, splitter)

    logger.info("=" * 60)
    logger.info("STAGE 2: Final model training")
    logger.info("=" * 60)
    model = trainer.train_final_model(train_df, best_params, splitter)

    # -- Holdout evaluation ---
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION on holdout")
    logger.info("=" * 60)
    metrics = trainer.evaluate_holdout(model, holdout_df)

    # -- Save ---
    trainer.save(model, config, metrics)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
