"""Entrypoint for the NYC Taxi fare prediction pipeline."""

from __future__ import annotations

import logging
import os

from taxi_pipeline.config import Config
from taxi_pipeline.data_processor import DataProcessor
from taxi_pipeline.data_utils import resolve_data
from taxi_pipeline.splitter import TimeSplitter
from taxi_pipeline.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TaxiPipeline")


def run_pipeline() -> None:
    """Orchestrate the full training pipeline."""
    months = ["01", "02", "03"]
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_paths = resolve_data(months, dest_dir=data_dir)

    config = Config(data_paths=data_paths)

    # Data processing
    processor = DataProcessor(config)
    df = processor.load_and_optimize()
    df = processor.sanity_check_and_clean(df)
    df = processor.feature_engineering(df)

    df = df.sort(config.datetime_column)
    date_min = df[config.datetime_column].min()
    date_max = df[config.datetime_column].max()
    logger.info("Data range: from %s to %s", date_min, date_max)

    # Chronological split
    splitter = TimeSplitter(config)
    train_df, holdout_df = splitter.split_holdout(df)

    # Training
    trainer = ModelTrainer(config)

    logger.info("=" * 60)
    logger.info("STAGE 1: Hyperparameter tuning")
    logger.info("=" * 60)
    best_params = trainer.tune_hyperparameters(train_df, splitter)

    logger.info("=" * 60)
    logger.info("STAGE 2: Final model training")
    logger.info("=" * 60)
    model = trainer.train_final_model(train_df, best_params, splitter)

    # Final evaluation
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION on holdout")
    logger.info("=" * 60)
    metrics = trainer.evaluate_holdout(model, holdout_df)

    # Save
    trainer.save(model, config, metrics)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
