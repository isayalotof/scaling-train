from __future__ import annotations

import os

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
