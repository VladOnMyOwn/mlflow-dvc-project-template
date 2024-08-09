import os
import tempfile
from pathlib import Path
from typing import Union

# import dvc.api
import joblib
import mlflow
import pandas as pd
from loguru._logger import Logger
from xgboost import Booster, XGBClassifier

from config.core import config


def get_last_run(experiment_id: str, run_name: str,
                 logger: Logger) -> pd.Series:

    last_run = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and status = 'FINISHED'",  # noqa
        order_by=["start_time DESC"]
    ).loc[0, :]

    if last_run.empty:
        message = f"Run {run_name} was not found"
        logger.error(message)
        raise Exception(message)

    return last_run


def get_run_by_id(experiment_id: str, run_id: str,
                  logger: Logger) -> pd.Series:

    run = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_id = '{run_id}' and status = 'FINISHED'"  # noqa
    ).loc[0, :]

    if run.empty:
        message = f"Run with id {run_id} was not found"
        logger.error(message)
        raise Exception(message)

    return run


def load_logged_data(
    run_id: str,
    tmp_path: Union[str, os.PathLike],
    dataset_name: str,
    logger: Logger,
    dst_dir: str = "datasets",
    log_usage: bool = False,
    **log_usage_kwargs
) -> pd.DataFrame:

    if isinstance(tmp_path, str):
        tmp_path = Path(tmp_path)

    with tempfile.TemporaryDirectory(dir=tmp_path) as tmpdir:
        logger.info("Created directory {} for downloading {} dataset".format(
            tmpdir, dataset_name))

        # Вариант выгрузки данных, если они залогированы как текстовый артефакт  # noqa
        # mlflow.artifacts.download_artifacts(
        #     run_id=last_prep_run["run_id"],
        #     artifact_path="datasets/train.csv",
        #     dst_path=tmpdir
        # )

        run = mlflow.get_run(run_id)
        dataset_input = [dsi for dsi in run.inputs.dataset_inputs
                         if dsi.dataset.name == dataset_name][0]
        dataset_source = mlflow.data.get_source(dataset_input)
        dataset_source.load(dst_path=os.path.join(tmpdir, dst_dir))

        # File name must match the dataset name
        data = pd.read_csv(os.path.join(
            tmpdir,
            f"{dst_dir}/{dataset_name}.{config.project.datasets_file_format}")
        )

    logger.info(f"Dataset {dataset_name} loaded into memory")

    if log_usage:
        dataset = mlflow.data.from_pandas(
            data,
            name=dataset_input.dataset.name,
            targets=log_usage_kwargs["targets"],
            source=dataset_source,
            digest=dataset_input.dataset.digest
        )
        mlflow.log_input(dataset, context=log_usage_kwargs["context"])

    return data


def log_xgboost_model(
    model: Union[Booster, XGBClassifier],
    artifact_path: str,
    input_example: pd.DataFrame,
    prediction_example: str,
    model_name: str,
    model_alias: str,
    mlflow_model_save_format: str,
    local_model_save_format: str,
    local_models_path: Union[str, os.PathLike],
    model_name_suffix: str = ""
) -> None:

    if isinstance(local_models_path, str):
        local_models_path = Path(local_models_path)

    client = mlflow.MlflowClient()

    # Log and register model
    model_info = mlflow.xgboost.log_model(
        model,
        artifact_path=artifact_path,
        input_example=input_example,
        registered_model_name=model_name + model_name_suffix,
        model_format=mlflow_model_save_format
    )
    model_version = model_info.registered_model_version
    client.set_registered_model_alias(
        model_name + model_name_suffix,
        version=model_version,
        alias=model_alias
    )

    # Log predictions example
    mlflow.log_text(
        prediction_example,
        artifact_file=f"{artifact_path}/predictions_example.json"
    )

    # Save locally and log artifacts
    local_model_file = "{}_v{}.{}".format(
        model_name + model_name_suffix,
        model_version,
        local_model_save_format
    )
    model.save_model(local_models_path / local_model_file)
    mlflow.log_artifact(
        local_models_path / local_model_file,
        artifact_path=artifact_path
    )

    joblib.dump(
        model, local_models_path /
        f"{model_name + model_name_suffix}_v{model_version}.pkl"
    )
    mlflow.log_artifact(
        local_models_path /
        f"{model_name + model_name_suffix}_v{model_version}.pkl",
        artifact_path=artifact_path
    )


# def load_versioned_data():

#     with dvc.api.open(
#             path=,
#             rev="v1",
#             remote=,
#             remote_config=) as f:
#     ...
