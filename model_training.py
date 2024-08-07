import logging
import os
import sys
import tempfile
import warnings

import mlflow
import pandas as pd
import xgboost as xgb
from loguru import logger

from config.core import config


# set up logging
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


if __name__ == "__main__":

    logger.info("Model training started")

    mlflow.set_tracking_uri(config.project.tracking_server_uri)
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

        # get last finished run for data preprocessing
        last_prep_run_id = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.mlflow.runName = 'Data_Preprocessing' and status = 'FINISHED'",  # noqa
            order_by=["start_time DESC"]
        ).loc[0, "run_id"]

        # download train and test data from last run
        with tempfile.TemporaryDirectory() as tmpdir:  # TODO: указать путь
            logger.info(
                f"Created directory {tmpdir} for downloading datasets")

            last_prep_run = mlflow.get_run(last_prep_run_id)
            dataset_inputs = [
                dsi for dsi in last_prep_run.inputs.dataset_inputs
                if dsi.dataset.name in ["train", "test"]
            ]
            for dataset_input in dataset_inputs:
                dataset_source = mlflow.data.get_source(dataset_input)
                dataset_source.load(dst_path=os.path.join(
                    tmpdir, config.project.artifacts_datasets_dir))

            train = pd.read_csv(os.path.join(
                tmpdir, f"{config.project.artifacts_datasets_dir}/train.csv"))
            test = pd.read_csv(os.path.join(
                tmpdir, f"{config.project.artifacts_datasets_dir}/test.csv"))

            # log datasets
            for dataset_input in dataset_inputs:
                dataset_source = mlflow.data.get_source(dataset_input)
                dataset = mlflow.data.from_pandas(
                    train if dataset_input.dataset.name == "train" else test,
                    name=dataset_input.dataset.name,
                    targets="target",
                    source=dataset_source,
                    digest=dataset_input.dataset.digest
                )
                mlflow.log_input(
                    dataset, context=f"{dataset_input.dataset.name}ing")

        # convert to DMatrix format
        features = [i for i in train.columns if i != "target"]
        dtrain = xgb.DMatrix(data=train[features], label=train["target"])
        dtest = xgb.DMatrix(data=test[features], label=test["target"])

        # get last finished run for hyperparameters tuning
        last_tuning_run = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.mlflow.runName = 'Hyperparameters_Search' and status = 'FINISHED'",  # noqa
            order_by=["start_time DESC"]
        ).loc[0, :]

        # get best params
        params = {col.split(".")[1]: last_tuning_run[col]
                  for col in last_tuning_run.index if "params" in col}
        params.update(eval_metric=config.model.params_eval_metrics)
        del params["n-trials"]

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

        # mlflow.log_metric(
        #     f"test_{config.model.params_tuning_metric}",
        #     model.best_score
        # )  # already logged by autologging as best iteration (and appended as last iteration to all plots)  # noqa
        logger.info("Best iteration test_{}: {}".format(
            config.model.params_tuning_metric, model.best_score))

        # log and register model
        input_example = test.loc[0:10, features]
        predictions_example = pd.DataFrame(
            model.predict(xgb.DMatrix(input_example)),
            columns=["predictions"]
        )
        mlflow.xgboost.log_model(
            model,
            artifact_path="booster",
            input_example=input_example,
            registered_model_name=config.model.model_name,
            model_format=config.model.model_save_format,
        )
        mlflow.log_text(
            predictions_example.to_json(orient="split", index=False),
            artifact_file="booster/predictions_example.json"
        )

        # log and register model as sklearn compatible classifier
        params.update(num_boost_round=model.best_iteration)
        skl_model = xgb.XGBClassifier(**params)
        skl_model.fit(train[features], train["target"])
        mlflow.xgboost.log_model(
            skl_model,
            artifact_path="sklearn",
            input_example=input_example,
            registered_model_name=config.model.model_name + "_sklearn",
            model_format=config.model.sklearn_save_format
        )

        logger.info("Model training finished")
