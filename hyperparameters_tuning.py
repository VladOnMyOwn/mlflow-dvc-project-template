import argparse
import logging
import sys
import warnings
from typing import Literal

import mlflow
import optuna
import xgboost as xgb
from loguru import logger
from xgboost.callback import TrainingCallback

from config.core import PROJECT_ROOT, config
from utils import get_last_run, get_run_by_id, load_logged_data


# set up logging
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
optuna.logging.set_verbosity(optuna.logging.ERROR)


# custom callback for logging metrics
class LoggingCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log) -> Literal[False]:
        # считается средняя метрика для train и test
        for data_name, metric_history in evals_log.items():
            for metric_name, metric_vals in metric_history.items():
                mlflow.log_metric(
                    f"cv_{data_name}_" + metric_name,
                    metric_vals[-1][0],  # [0], т.к. считается среднее и станд. отклон.  # noqa
                    step=epoch
                )
        return False


# define an objective function for optuna
def objective(trial) -> float:
    global dtrain

    params = {
        "objective": trial.suggest_categorical("objective", ["binary:logistic"]),  # обернуто в suggest, чтобы тоже логировалось и отдавалось на выходе  # noqa
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "alpha": trial.suggest_float("alpha", 0.001, 0.05),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
        "num_boost_round": trial.suggest_int("num_boost_round", 30, 300),
    }

    with mlflow.start_run(nested=True):

        mlflow.log_params(params)
        params.update(eval_metric=config.model.params_eval_metrics)
        cv_results = xgb.cv(  # performs CV at each boosting iteration
            params,
            dtrain,
            num_boost_round=params["num_boost_round"],
            nfold=config.model.cv_n_folds,
            early_stopping_rounds=max(
                int(params["num_boost_round"] * config.model.early_stopping_heuristic),  # noqa
                1
            ),
            callbacks=[LoggingCallback()],
            verbose_eval=False
        )

        early_stopping = len(cv_results) < params["num_boost_round"]
        if early_stopping:
            mlflow.set_tags({
                "early_stopping": early_stopping,
                "best_step": len(cv_results) - 1
            })

        objective_score = cv_results[f"test-{config.model.params_tuning_metric}-mean"].iloc[-1]  # берется CV-метрика с последней итерации бустинга  # noqa

        if config.model.additional_metrics is not None:
            additional_metrics = {}
            for dataset in ["train", "test"]:
                additional_metrics.update({
                    f"cv_{dataset}_{metric_name}": eval(
                        computation["formula"].format(
                            cv_results[f"{dataset}-{computation['source']}-mean"].iloc[-1]  # noqa
                        )
                    )
                    for metric_name, computation in config.model.additional_metrics.items()  # noqa
                })
            mlflow.log_metrics(additional_metrics)

        trial_log = ", ".join(
            f"cv_{m}: {round(cv_results[f'test-{m}-mean'].iloc[-1], config.project.logging_precision)}"  # noqa
            for m in config.model.params_eval_metrics)
        logger.info(
            f"Attempt: {trial.number}, Early stopping: {early_stopping} | {trial_log}")  # noqa

        return objective_score


if __name__ == "__main__":

    # get arguments if running not in ipykernel
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-run-id", default="", type=str)
    parser.add_argument(
        "--n-trials", default=config.model.params_tuning_n_trials, type=int)
    cmd_args = parser.parse_args()
    DATA_RUN_ID = cmd_args.data_run_id
    N_TRIALS = cmd_args.n_trials

    logger.info(f"Hyperparameters tuning started with {N_TRIALS} trials")

    mlflow.set_tracking_uri(config.project.tracking_server_uri)

    with mlflow.start_run(log_system_metrics=True) as run:

        # get experiment id
        experiment_id = run.info.experiment_id

        if not DATA_RUN_ID:
            # get last finished run for data preprocessing
            data_run = get_last_run(
                experiment_id, "Data_Preprocessing", logger)
        else:
            # get data preprocessing run with specified run id
            data_run = get_run_by_id(
                experiment_id, DATA_RUN_ID, logger)

        # download train data from last run
        tmpdir_path = PROJECT_ROOT / "tmp"
        tmpdir_path.mkdir(exist_ok=True, parents=True)
        train = load_logged_data(
            run_id=data_run["run_id"],
            tmp_path=tmpdir_path,
            dataset_name="train",
            logger=logger,
            dst_dir=config.project.artifacts_datasets_dir,
            log_usage=True,
            targets="target",
            context="finetuning"
        )

        # convert to DMatrix format
        features = [i for i in train.columns if i != "target"]
        dtrain = xgb.DMatrix(data=train[features], label=train["target"])

        logger.info("Starting optuna study with objective: {} -> {}".format(
            config.model.params_tuning_metric, config.model.params_tuning_direction))  # noqa

        study = optuna.create_study(
            direction=config.model.params_tuning_direction)
        study.optimize(objective, n_trials=N_TRIALS)
        best_trial = study.best_trial

        mlflow.log_params(best_trial.params)
        logger.success(
            f"Optimization finished, best params: {best_trial.params}")

        mlflow.log_metric(f"cv_test_{config.model.params_tuning_metric}",
                          best_trial.value)  # CV-метрика с последней итерации
        logger.info("Best trial {}: {}".format(
            config.model.params_tuning_metric,
            round(best_trial.value, config.project.logging_precision)))
