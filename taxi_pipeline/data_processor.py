"""Data loading, type optimization, cleaning, and feature engineering."""

from __future__ import annotations

import logging

import polars as pl

from taxi_pipeline.config import Config

logger = logging.getLogger("TaxiPipeline")


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

        float64_cols = [c for c in df.columns if df[c].dtype == pl.Float64]
        if float64_cols:
            df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols])

        int64_cols = [c for c in df.columns if df[c].dtype == pl.Int64]
        if int64_cols:
            df = df.with_columns([pl.col(c).cast(pl.Int32) for c in int64_cols])

        mem_after = df.estimated_size("mb")
        logger.info(
            "Memory: %.2f MB -> %.2f MB (saved %.1f%%)",
            mem_before, mem_after, (1 - mem_after / mem_before) * 100,
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

        critical_cols = [
            self._config.target_column,
            "trip_distance",
            self._config.datetime_column,
            self._config.dropoff_column,
        ]
        df = df.drop_nulls(subset=[c for c in critical_cols if c in df.columns])

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
            rows_before, rows_after,
            rows_before - rows_after,
            (rows_before - rows_after) / rows_before * 100,
        )

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

        Features: pickup_hour, pickup_day, pickup_month, pickup_weekday,
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

        df = df.filter(pl.col("trip_duration_seconds") > 0)

        logger.info(
            "Feature engineering complete. Columns: %d, Rows: %d",
            len(df.columns), df.height,
        )
        return df
