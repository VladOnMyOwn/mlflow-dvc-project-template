import sys
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

sys.path.append("..")
from mlproject import __file__  # noqa


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"


class ProjectConfig(BaseModel):
    artifacts_destination: str
    tracking_server_uri: str
    artifacts_datasets_dir: str
    logging_precision: int


class ModelConfig(BaseModel):
    model_save_dir: str
    model_name: str
    model_type: str
    champion_model_alias: str
    load_by_alias: bool
    mlflow_model_save_format: str
    local_model_save_format: str
    default_test_size: float
    params_tuning_n_trials: int
    target_name: str
    cv_n_folds: int
    params_eval_metrics: List[str]
    params_tuning_metric: str
    params_tuning_direction: str
    additional_metrics: Optional[Dict[str, Dict[str, str]]]
    early_stopping_heuristic: float
    importance_types: List[str]


class Config(BaseModel):
    project: ProjectConfig
    model: ModelConfig


def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config


def create_and_validate_config(
        parsed_config: Optional[YAML] = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        project=ProjectConfig(**parsed_config.data),
        model=ModelConfig(**parsed_config.data)
    )

    return _config


config = create_and_validate_config()
