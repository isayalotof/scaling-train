from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    data_paths: list[str] = field(default_factory=list)
    datetime_column: str = "tpep_pickup_datetime"
    dropoff_column: str = "tpep_dropoff_datetime"
    target_column: str = "fare_amount"

    n_cv_splits: int = 5
    n_trials: int = 30
    tuning_sample_fraction: float = 0.2
    holdout_fraction: float = 0.15

    high_value_threshold: float = 500.0
    max_passengers: int = 10
    min_fare: float = 0.0
    max_fare_hard: float = 5000.0
    max_distance: float = 500.0

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

    model_save_path: str = "model.pkl"
    config_save_path: str = "config.json"
    random_seed: int = 42
