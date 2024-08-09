import argparse
import sys
import warnings

import mlflow
import pandas as pd
from loguru import logger
from sklearn import datasets
from sklearn.model_selection import train_test_split

from config.core import config


# set up logging
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
    TEST_SIZE = parser.parse_args().test_size

    logger.info(f"Data preprocessing started with test size: {TEST_SIZE}")

    mlflow.set_tracking_uri(config.project.tracking_server_uri)

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
    mlflow.log_text(
        train.to_csv(index=False),
        "{}/{}.{}".format(
            config.project.artifacts_datasets_dir,
            config.project.train_dataset_name,
            config.project.datasets_file_format
        )  # will be logged into experiment_id/run_id/artifacts/datasets
    )
    # вместо этого можно делать log_artifact для любого формата файла, но
    # не желательно, т.к. данные будут дублироваться для каждого запуска
    # или записывать файл в удаленное/локальное хранилище
    dataset_source_link = mlflow.get_artifact_uri(
        "{}/{}.{}".format(
            config.project.artifacts_datasets_dir,
            config.project.train_dataset_name,
            config.project.datasets_file_format
        )
    ).replace(
        "mlflow-artifacts:/",
        config.project.artifacts_destination
    )  # mlflow-artifacts:/experiment_id/run_id/artifacts/datasets/train.csv throws an error  # noqa
    dataset = mlflow.data.from_pandas(
        train,
        name=config.project.train_dataset_name,
        targets=config.model.target_name,
        source=dataset_source_link  # можно указать путь к файлу (в т.ч. в s3)
    )
    mlflow.log_input(dataset, context="preprocessing")

    test = X_test.assign(**{config.model.target_name: y_test})
    mlflow.log_text(
        test.to_csv(index=False),
        "{}/{}.{}".format(
            config.project.artifacts_datasets_dir,
            config.project.test_dataset_name,
            config.project.datasets_file_format
        )
    )
    dataset_source_link = mlflow.get_artifact_uri(
        "{}/{}.{}".format(
            config.project.artifacts_datasets_dir,
            config.project.test_dataset_name,
            config.project.datasets_file_format
        )
    ).replace(
        "mlflow-artifacts:/",
        config.project.artifacts_destination
    )
    dataset = mlflow.data.from_pandas(
        test,
        name=config.project.test_dataset_name,
        targets=config.model.target_name,
        source=dataset_source_link
    )
    mlflow.log_input(dataset, context="preprocessing")

    logger.info("Data preprocessing finished")
