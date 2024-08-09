import argparse
import logging
import sys
import warnings

import mlflow
import pandas as pd
from loguru import logger

from config.core import config, PROJECT_ROOT


warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=str)
    eval_dataset_src = parser.parse_args().eval_dataset

    eval_dataset = pd.read_csv(PROJECT_ROOT / eval_dataset_src)

    logger.info("Evaluation started")

    mlflow.set_tracking_uri(config.project.tracking_server_uri)

    with mlflow.start_run() as run:

        eval_dataset = mlflow.data.from_pandas(
            eval_dataset,
            name="eval",
            targets=config.model.target_name,
            source=PROJECT_ROOT / eval_dataset_src
        )
        # mlflow.log_input(eval_dataset, context="evaluation")  # will be logged by .evaluate method  # noqa

        model_uri = f"models:/{config.model.model_name + '_sklearn'}"
        if config.model.load_by_alias:
            model_uri += f"@{config.model.champion_model_alias}"
        else:
            latest_version = mlflow.MlflowClient().get_registered_model(
                config.model.model_name + '_sklearn').latest_versions[0].version  # noqa
            model_uri += f"/{latest_version}"

        mlflow.evaluate(
            model=model_uri,
            model_type=config.model.model_type,
            data=eval_dataset,
            dataset_path=eval_dataset_src,
            evaluator_config={"pos_label": 1},
            # extra_metrics=
        )  # не считает все для booster, т.к. его predict выдает вероятность
        # метрики на вероятностях считает только для sklearn
        # TODO: сделать альтернативную валидацию для booster

        logger.success("Evaluation finished")
