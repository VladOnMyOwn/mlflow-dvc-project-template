from os import PathLike
from pathlib import Path
from typing import Union

import joblib
import mlflow
from pandas import DataFrame
from xgboost import Booster, XGBClassifier


def log_xgboost_model(
    model: Union[Booster, XGBClassifier],
    artifact_path: str,
    input_example: DataFrame,
    prediction_example: str,
    model_name: str,
    model_alias: str,
    mlflow_model_save_format: str,
    local_model_save_format: str,
    local_models_path: Union[str, PathLike],
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
