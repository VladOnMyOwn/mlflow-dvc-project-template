import argparse
import logging
import sys
import warnings

import mlflow
import pandas as pd
import xgboost as xgb
from loguru import logger

from config.core import PROJECT_ROOT, config
from utils import (get_last_run, get_run_by_id, load_versioned_data,
                   log_xgboost_model)


warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-run-id", default="", type=str)
    parser.add_argument("--tuning-run-id", default="", type=str)
    parser.add_argument("--data-version", default="", type=str)
    cmd_args = parser.parse_args()
    DATA_RUN_ID = cmd_args.data_run_id
    DVC_REVISION = cmd_args.data_version
    PARAMS_RUN_ID = cmd_args.tuning_run_id

    logger.info("Model training started")

    mlflow.set_tracking_uri(config.project.tracking_uri)
    mlflow.xgboost.autolog(
        importance_types=config.model.importance_types,
        log_datasets=False,  # will be logged manually for better control
        log_models=False,  # will be logged manually for better control
    )
    # for more: https://mlflow.org/docs/latest/python_api/mlflow.xgboost.html#mlflow.xgboost.autolog  # noqa

    with mlflow.start_run(log_system_metrics=True) as run:

        experiment_id = run.info.experiment_id

        run_id = run.info.run_id
        logger.info(f"Starting MLflow run: {run_id}")

        if not DATA_RUN_ID:
            # get last finished run for data preprocessing
            if not DVC_REVISION:
                data_run = get_last_run(
                    experiment_id, "Data_Preprocessing", logger)
            else:
                # filter by dataset_version tag
                data_run = get_last_run(
                    experiment_id, "Data_Preprocessing", logger,
                    dataset_version=DVC_REVISION)
        else:
            # get data preprocessing run with specified run id
            data_run = get_run_by_id(
                experiment_id, DATA_RUN_ID, logger)

        # download train and test data
        train = load_versioned_data(
            run_id=data_run["run_id"],
            dataset_name=config.project.train_dataset_name,
            logger=logger,
            log_usage=True,
            targets=config.model.target_name,
            context="training"
        )
        test = load_versioned_data(
            run_id=data_run["run_id"],
            dataset_name=config.project.test_dataset_name,
            logger=logger,
            log_usage=True,
            targets=config.model.target_name,
            context="testing"
        )

        # convert to DMatrix format
        features = [i for i in train.columns if i != config.model.target_name]
        dtrain = xgb.DMatrix(
            data=train[features],
            label=train[config.model.target_name]
        )
        dtest = xgb.DMatrix(
            data=test[features],
            label=test[config.model.target_name]
        )

        if not PARAMS_RUN_ID:
            # get last finished parent run for hyperparameters tuning
            tuning_run = get_last_run(
                experiment_id, "Hyperparameters_Search", logger)
        else:
            # get hyperparameters tuning run with specified run id
            tuning_run = get_run_by_id(
                experiment_id, PARAMS_RUN_ID, logger)

        # get best params
        params = {
            col.split(".")[1]: tuning_run[col]
            for col in tuning_run.index if (
                ("params" in col) and (all(
                    p not in col
                    for p in ["n-trials", "data-run-id", "data-version"]
                ))
            )
        }
        params.update(eval_metric=config.model.params_eval_metrics)

        mlflow.log_params(params)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=int(params["num_boost_round"]),
            evals=[(dtest, "test")],
            verbose_eval=False,
            early_stopping_rounds=max(
                int(int(params["num_boost_round"]) * config.model.early_stopping_heuristic),  # noqa
                1
            )
        )

        logger.info("Best iteration test_{}: {}".format(
            config.model.params_tuning_metric, model.best_score))

        local_models_path = PROJECT_ROOT / config.model.save_dir
        local_models_path.mkdir(exist_ok=True, parents=True)

        # log and register model
        input_example = test.loc[0:10, features]
        predictions_example = pd.DataFrame(
            model.predict(xgb.DMatrix(input_example)),
            columns=["predictions"]
        )
        log_xgboost_model(
            model,
            artifact_path="booster",
            input_example=input_example,
            prediction_example=predictions_example.to_json(
                orient="split", index=False),
            model_name=config.model.name,
            model_alias=config.model.champion_alias,
            mlflow_model_save_format=config.model.mlflow_save_format,
            local_model_save_format=config.model.local_save_format,
            local_models_path=local_models_path
        )

        # TODO: add logging custom artifacts:
        # ROC plot, PRC plot, calibration curve

        # log and register model as sklearn compatible classifier
        params.update(num_boost_round=model.best_iteration)
        skl_model = xgb.XGBClassifier(**params)
        skl_model.fit(train[features], train[config.model.target_name])
        predictions_example = pd.DataFrame(
            skl_model.predict_proba(input_example)[:, 1],
            columns=["predictions"]
        )
        log_xgboost_model(
            skl_model,
            artifact_path="sklearn",
            input_example=input_example,
            prediction_example=predictions_example.to_json(
                orient="split", index=False),
            model_name=config.model.name,
            model_alias=config.model.champion_alias,
            mlflow_model_save_format=config.model.mlflow_save_format,
            local_model_save_format=config.model.local_save_format,
            local_models_path=local_models_path,
            model_name_suffix="_sklearn"
        )

        logger.success("Model training finished")
