import os
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from typing import List, Union
from urllib.parse import unquote, urlparse

import dvc.api
import joblib
import mlflow
import pandas as pd
from loguru._logger import Logger
from xgboost import Booster, XGBClassifier

from config.core import PROJECT_ROOT, config


def get_last_run(experiment_id: str, run_name: str,
                 logger: Logger, **filter_kwargs) -> pd.Series:

    filters = f"tags.mlflow.runName = '{run_name}' and status = 'FINISHED'"
    if "dataset_version" in filter_kwargs.keys():
        filters += " and tags.`dataset_version` = '{}'".format(
            filter_kwargs['dataset_version'])

    last_run = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filters,
        order_by=["start_time DESC"]
    )

    if last_run.empty:
        message = f"Run {run_name} was not found"
        if "dataset_version" in filter_kwargs.keys():
            message += f", filter kwargs: {filter_kwargs.__repr__()}"
        logger.error(message)
        raise Exception(message)

    return last_run.loc[0, :]


def get_run_by_id(experiment_id: str, run_id: str,
                  logger: Logger) -> pd.Series:

    run = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_id = '{run_id}' and status = 'FINISHED'"  # noqa
    )

    if run.empty:
        message = f"Run with id {run_id} was not found"
        logger.error(message)
        raise Exception(message)

    return run.loc[0, :]


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


def create_data_version(
    dataset_names: List[str],
    version: str,
    commit_message: str,
    logger: Logger,
    force_version: bool = False
) -> None:

    def _dvc_push(dataset_name: str) -> List[str]:
        dvc_push_data = [
            "dvc",
            "add",
            "{}/{}.{}".format(
                config.project.local_datasets_dir,
                dataset_name,
                config.project.datasets_file_format
            ),
            "--to-remote",
            "-r",
            config.project.dvc_remote_name
        ]
        return dvc_push_data

    def _git_stage(dataset_name: str) -> List[str]:
        git_stage_data = [
            "git",
            "add",
            "{}/{}.{}.dvc".format(
                config.project.local_datasets_dir,
                dataset_name,
                config.project.datasets_file_format
            )
        ]
        return git_stage_data

    def _git_tag() -> List[str]:
        git_tag_pointers = [
            "git",
            "tag",
            "-a",
            version,
            "-m",
            f"data: {commit_message}"
        ]
        if force_version:
            # force if exists
            git_tag_pointers += ["-f"]
        return git_tag_pointers

    def _git_push_tag() -> List[str]:
        git_push_pointers_tag = ["git", "push"]
        if force_version:
            # force if exists
            git_push_pointers_tag += ["-f"]
        git_push_pointers_tag += ["--tag"]
        return git_push_pointers_tag

    logger.info(f"Data versioning started: {version}")

    git_commit_pointers = [
        "git",
        "commit",
        "-m",
        f"data: {commit_message}"
    ]
    git_push_pointers = ["git", "push"]

    for dataset in dataset_names:
        subprocess.run(_dvc_push(dataset), cwd=PROJECT_ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    logger.info("Data were pushed to dvc remote")

    for dataset in dataset_names:
        subprocess.run(_git_stage(dataset), cwd=PROJECT_ROOT, check=True)
    logger.info("Data were added to git staging area")

    subprocess.run(git_commit_pointers, cwd=PROJECT_ROOT, check=True)
    subprocess.run(_git_tag(), cwd=PROJECT_ROOT, check=True)
    logger.info("Data pointers were commited locally")

    subprocess.run(git_push_pointers, cwd=PROJECT_ROOT, check=True)
    subprocess.run(_git_push_tag(), cwd=PROJECT_ROOT, check=True)
    logger.info("Data pointers were pushed to git repository")

    mlflow.set_tag("dataset_version", version)


def load_versioned_data(
    run_id: str,
    dataset_name: str,
    logger: Logger,
    log_usage: bool = False,
    **log_usage_kwargs
) -> pd.DataFrame:

    run = mlflow.get_run(run_id)
    version = run.data.tags["dataset_version"]
    dataset_input = [dsi for dsi in run.inputs.dataset_inputs
                     if dsi.dataset.name == dataset_name][0]
    dataset_uri = mlflow.data.get_source(dataset_input).to_dict()["uri"]
    dataset_src = unquote(urlparse(dataset_uri).path)

    contents = dvc.api.read(
        repo=PROJECT_ROOT.as_uri(),
        path=os.path.relpath(dataset_src, PROJECT_ROOT),
        rev=version,  # HEAD if None
        remote=config.project.dvc_remote_name,
        remote_config=config.storage.dict()
        if config.storage is not None else None,
        mode="r"  # rb
    )

    data = pd.read_csv(StringIO(contents))  # BytesIO

    logger.info(f"Dataset {dataset_name} loaded into memory")

    if log_usage:
        dataset = mlflow.data.from_pandas(
            data,
            name=dataset_input.dataset.name,
            targets=log_usage_kwargs["targets"],
            source=dataset_uri,
            digest=dataset_input.dataset.digest
        )
        mlflow.log_input(dataset, context=log_usage_kwargs["context"])
        mlflow.set_tag("dataset_version", version)

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
