from __future__ import annotations

import os

import mlflow

from taxi_pipeline.config import Config
from taxi_pipeline.data_processor import DataProcessor
from taxi_pipeline.data_utils import resolve_data
from taxi_pipeline.splitter import TimeSplitter
from taxi_pipeline.trainer import ModelTrainer


def run_pipeline() -> None:
    months = ["01", "02", "03"]
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_paths = resolve_data(months, dest_dir=data_dir)

    config = Config(data_paths=data_paths)

    mlruns_dir = os.path.join(data_dir, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    mlflow.set_experiment("taxi_fare_prediction")

    with mlflow.start_run():
        mlflow.log_params({
            "months": ",".join(months),
            "n_cv_splits": config.n_cv_splits,
            "n_trials": config.n_trials,
            "tuning_sample_fraction": config.tuning_sample_fraction,
            "holdout_fraction": config.holdout_fraction,
            "random_seed": config.random_seed,
            "high_value_threshold": config.high_value_threshold,
            "max_passengers": config.max_passengers,
            "min_fare": config.min_fare,
        })

        processor = DataProcessor(config)
        df = processor.load_and_optimize()
        df = processor.sanity_check_and_clean(df)
        df = processor.feature_engineering(df)
        df = df.sort(config.datetime_column)

        splitter = TimeSplitter(config)
        train_df, holdout_df = splitter.split_holdout(df)

        trainer = ModelTrainer(config)
        best_params = trainer.tune_hyperparameters(train_df, splitter)
        model = trainer.train_final_model(train_df, best_params, splitter)
        metrics = trainer.evaluate_holdout(model, holdout_df)
        trainer.save(model, config, metrics)


if __name__ == "__main__":
    run_pipeline()
