import mlflow
import pandas as pd
from loguru._logger import Logger


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
