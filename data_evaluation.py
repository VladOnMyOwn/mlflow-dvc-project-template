import sys
import os
import argparse
import warnings
import logging
import mlflow
import pandas as pd
from loguru import logger

from config.core import config


warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=str)
    eval_dataset_src = parser.parse_args().eval_dataset

    eval_dataset = pd.read_csv(eval_dataset_src)

    logger.info("Evaluation started")

    mlflow.set_tracking_uri(config.project.tracking_server_uri)

    with mlflow.start_run() as run:

        eval_dataset = mlflow.data.from_pandas(
            eval_dataset,
            name="eval",
            targets="target",
            source=eval_dataset_src
        )
        # mlflow.log_input(eval_dataset, context="evaluation")  # will be logged by .evaluate method  # noqa

        latest_version = mlflow.MlflowClient().get_registered_model(
            config.model.model_name).latest_versions[0].version
        mlflow.evaluate(
            model=f"models:/{config.model.model_name}/{latest_version}",
            model_type=config.model.model_type,
            data=eval_dataset,
            dataset_path=eval_dataset_src,
            evaluator_config={"pos_label": 1}
        )
