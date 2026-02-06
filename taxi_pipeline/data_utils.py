from __future__ import annotations

import calendar
import os
import subprocess

import numpy as np
import polars as pl

_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-{}.parquet"


def download_data(months: list[str], dest_dir: str = ".") -> list[str]:
    paths: list[str] = []
    for month in months:
        filename = f"yellow_tripdata_2023-{month}.parquet"
        filepath = os.path.join(dest_dir, filename)
        if not os.path.exists(filepath):
            url = _BASE_URL.format(month)
            subprocess.run(["wget", "-q", url, "-O", filepath], check=True)
        paths.append(filepath)
    return paths


def generate_synthetic_data(
    months: list[str],
    dest_dir: str = ".",
    rows_per_month: int = 100_000,
    seed: int = 42,
) -> list[str]:
    rng = np.random.default_rng(seed)
    paths: list[str] = []
    location_ids = list(range(1, 264))

    for month_str in months:
        month = int(month_str)
        year = 2023
        days_in_month = calendar.monthrange(year, month)[1]
        n = rows_per_month

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

        trip_distance = rng.lognormal(mean=1.0, sigma=0.7, size=n).clip(0.1, 80)

        base_speed_mph = rng.normal(12, 4, size=n).clip(3, 40)
        duration_seconds = ((trip_distance / base_speed_mph) * 3600).astype(int).clip(60, 7200)
        dropoffs = pickups + duration_seconds.astype("timedelta64[s]")

        fare_amount = 3.50 + 2.50 * trip_distance + rng.normal(0, 1.5, size=n)
        fare_amount = fare_amount.clip(2.5, 300)

        anomaly_idx = rng.choice(n, size=int(n * 0.02), replace=False)
        half = len(anomaly_idx) // 2
        fare_amount[anomaly_idx[:half]] = rng.uniform(-50, -0.01, size=half)
        trip_distance_arr = trip_distance.copy()
        trip_distance_arr[anomaly_idx[half:]] = 0.0

        passenger_count = rng.choice(
            [0, 1, 2, 3, 4, 5, 6, 11], size=n,
            p=[0.01, 0.70, 0.12, 0.06, 0.04, 0.03, 0.03, 0.01],
        ).astype(np.float64)
        pu_loc = rng.choice(location_ids, size=n)
        do_loc = rng.choice(location_ids, size=n)
        payment_type = rng.choice([1, 2, 3, 4], size=n, p=[0.6, 0.3, 0.05, 0.05])
        vendor_id = rng.choice([1, 2], size=n)
        ratecode_id = rng.choice(
            [1, 2, 3, 4, 5, 99], size=n,
            p=[0.9, 0.04, 0.02, 0.02, 0.01, 0.01],
        ).astype(np.float64)

        extra = rng.choice([0.0, 0.5, 1.0, 2.5], size=n, p=[0.3, 0.3, 0.2, 0.2])
        mta_tax = np.full(n, 0.5)
        tip_upper = np.maximum(fare_amount * 0.3, 0.01)
        tip_amount = np.where(payment_type == 1, rng.uniform(0, tip_upper), 0.0)
        tolls_amount = np.where(rng.random(n) < 0.05, rng.uniform(5, 15, size=n), 0.0)
        improvement_surcharge = np.full(n, 1.0)
        congestion_surcharge = rng.choice([0.0, 2.5], size=n, p=[0.3, 0.7])
        airport_fee = rng.choice([0.0, 1.25], size=n, p=[0.85, 0.15])
        total_amount = (
            fare_amount + extra + mta_tax + tip_amount
            + tolls_amount + improvement_surcharge
            + congestion_surcharge + airport_fee
        )

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

        null_idx = rng.choice(n, size=int(n * 0.023), replace=False)
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
        paths.append(filepath)

    return paths


def resolve_data(months: list[str], dest_dir: str = ".") -> list[str]:
    all_exist = all(
        os.path.exists(os.path.join(dest_dir, f"yellow_tripdata_2023-{m}.parquet"))
        for m in months
    )
    if all_exist:
        return [
            os.path.join(dest_dir, f"yellow_tripdata_2023-{m}.parquet")
            for m in months
        ]

    try:
        return download_data(months, dest_dir)
    except Exception:
        return generate_synthetic_data(months, dest_dir)
