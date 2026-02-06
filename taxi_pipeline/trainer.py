from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import asdict
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from taxi_pipeline.config import Config
from taxi_pipeline.splitter import TimeSplitter


class ModelTrainer:
    def __init__(self, config: Config) -> None:
        self._config = config

    def _prepare_arrays(
        self, df: pl.DataFrame
    ) -> tuple[Any, np.ndarray, list[str]]:
        available_features = [
            c for c in self._config.feature_columns if c in df.columns
        ]
        X = df.select(available_features).to_pandas()
        y = df.select(self._config.target_column).to_numpy().ravel()
        return X, y, available_features

    def _detect_gpu(self) -> str:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return "gpu"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "cpu"

    def tune_hyperparameters(
        self, train_df: pl.DataFrame, splitter: TimeSplitter
    ) -> dict[str, Any]:
        sample_size = int(train_df.height * self._config.tuning_sample_fraction)
        sample_df = train_df.head(sample_size)

        X_sample, y_sample, _ = self._prepare_arrays(sample_df)
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

            return float(np.mean(rmse_scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            study_name="taxi_fare_tuning",
        )
        study.optimize(
            objective,
            n_trials=self._config.n_trials,
            show_progress_bar=True,
        )

        return study.best_trial.params

    def train_final_model(
        self,
        train_df: pl.DataFrame,
        best_params: dict[str, Any],
        splitter: TimeSplitter,
    ) -> lgb.LGBMRegressor:
        X_full, y_full, _ = self._prepare_arrays(train_df)
        device = self._detect_gpu()

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

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        return model

    def evaluate_holdout(
        self, model: lgb.LGBMRegressor, holdout_df: pl.DataFrame
    ) -> dict[str, float]:
        X_hold, y_hold, _ = self._prepare_arrays(holdout_df)
        y_pred = model.predict(X_hold)

        return {
            "rmse": float(np.sqrt(mean_squared_error(y_hold, y_pred))),
            "mae": float(mean_absolute_error(y_hold, y_pred)),
            "r2": float(r2_score(y_hold, y_pred)),
        }

    def save(
        self, model: lgb.LGBMRegressor, config: Config, metrics: dict[str, float]
    ) -> None:
        with open(config.model_save_path, "wb") as f:
            pickle.dump(model, f)

        config_dict = asdict(config)
        config_dict["holdout_metrics"] = metrics
        with open(config.config_save_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
