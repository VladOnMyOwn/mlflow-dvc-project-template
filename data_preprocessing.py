import argparse
import os
import sys
import warnings
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger
from sklearn import datasets
from sklearn.model_selection import train_test_split

from config.core import PROJECT_ROOT, config
from utils import create_data_version


warnings.filterwarnings("ignore")
logger.remove()
logger.add(
    sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def get_cancer_df():
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target)
    logger.info("Cancer data downloaded")
    return X, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-size",
        default=config.model.default_test_size,
        type=float
    )
    parser.add_argument("--dvc-rev", default="v1.0", type=str)
    parser.add_argument("--dvc-message", default="", type=str)
    cmd_args = parser.parse_args()
    TEST_SIZE = cmd_args.test_size
    DVC_REVISION = cmd_args.dvc_rev
    DVC_MESSAGE = cmd_args.dvc_message

    logger.info(f"Data preprocessing started with test size: {TEST_SIZE}")

    mlflow.set_tracking_uri(config.project.tracking_uri)

    # download cancer dataset
    X, y = get_cancer_df()

    # compute additional features
    X["additional_feature"] = X["mean symmetry"] / X["mean texture"]
    logger.info("Additional features generated")

    # log dataset size and features count
    mlflow.log_metrics({
        "full_sample_size": X.shape[0],
        "features_count": X.shape[1]
    })

    # TODO: add logging custom artifacts:
    # correlation matrix, VIF table, WoE / IV tables, distibution plots

    # split dataset into train and test parts and log sizes to mlflow
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE)
    mlflow.log_metrics({
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0]
    })

    # log and register datasets
    train = X_train.assign(**{config.model.target_name: y_train})
    dataset_src = os.path.join(
        PROJECT_ROOT,
        config.project.local_datasets_dir,
        f"{config.project.train_dataset_name}.{config.project.datasets_file_format}"  # noqa
    )
    train.to_csv(dataset_src, index=False)
    dataset = mlflow.data.from_pandas(
        train,
        name=config.project.train_dataset_name,
        targets=config.model.target_name,
        source=Path(dataset_src).as_uri()
    )
    mlflow.log_input(dataset, context="preprocessing")

    test = X_test.assign(**{config.model.target_name: y_test})
    dataset_src = os.path.join(
        PROJECT_ROOT,
        config.project.local_datasets_dir,
        f"{config.project.test_dataset_name}.{config.project.datasets_file_format}"  # noqa
    )
    test.to_csv(dataset_src, index=False)
    dataset = mlflow.data.from_pandas(
        test,
        name=config.project.test_dataset_name,
        targets=config.model.target_name,
        source=Path(dataset_src).as_uri()
    )
    mlflow.log_input(dataset, context="preprocessing")

    create_data_version(
        dataset_names=[
            config.project.train_dataset_name,
            config.project.test_dataset_name
        ],
        version=DVC_REVISION,
        commit_message=DVC_MESSAGE,
        logger=logger,
        force_version=True
    )

    logger.success("Data preprocessing finished")
