"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Central configuration for the entire pipeline.

    Args:
        data_paths: List of parquet file paths to load.
        datetime_column: Column used for temporal ordering.
        dropoff_column: Dropoff timestamp column.
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
