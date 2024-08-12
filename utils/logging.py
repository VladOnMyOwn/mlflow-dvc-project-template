import os
from pathlib import Path
from typing import Union

import joblib
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from xgboost import Booster, XGBClassifier

from mlproject.config.core import config


def log_xgboost_model(
    model: Union[Booster, XGBClassifier],
    artifact_path: str,
    input_example: pd.DataFrame,
    prediction_example: str,
    model_name: str,
    model_alias: str,
    mlflow_save_format: str,
    local_save_format: str,
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
        model_format=mlflow_save_format
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
        local_save_format
    )
    model.save_model(local_models_path / local_model_file)
    mlflow.log_artifact(
        local_models_path / local_model_file,
        artifact_path=artifact_path
    )

    joblib.dump(
        model, local_models_path /
        "{}_v{}.{}".format(
            model_name + model_name_suffix,
            model_version,
            config.model.sklearn_save_format
        )
    )
    mlflow.log_artifact(
        local_models_path /
        "{}_v{}.{}".format(
            model_name + model_name_suffix,
            model_version,
            config.model.sklearn_save_format
        ),
        artifact_path=artifact_path
    )


def log_sklearn_model(
    model: Union[ClassifierMixin, RegressorMixin],
    artifact_path: str,
    input_example: pd.DataFrame,
    prediction_example: str,
    model_name: str,
    model_alias: str,
    local_models_path: Union[str, os.PathLike],
    model_name_suffix: str = ""
) -> None:

    if isinstance(local_models_path, str):
        local_models_path = Path(local_models_path)

    client = mlflow.MlflowClient()

    # Log and register model
    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        input_example=input_example,
        registered_model_name=model_name + model_name_suffix,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        pyfunc_predict_fn=config.model.sklearn_predict_fn
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
    joblib.dump(
        model, local_models_path /
        "{}_v{}.{}".format(
            model_name + model_name_suffix,
            model_version,
            config.model.sklearn_save_format
        )
    )
    mlflow.log_artifact(
        local_models_path /
        "{}_v{}.{}".format(
            model_name + model_name_suffix,
            model_version,
            config.model.sklearn_save_format
        ),
        artifact_path=artifact_path
    )
