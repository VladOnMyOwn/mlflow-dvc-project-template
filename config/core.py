import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import dotenv_values
from pydantic import BaseModel
from strictyaml import YAML, load

sys.path.append("..")
from mlproject import __file__  # noqa


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"
STORAGE_CONFIG_FILE_PATH = PROJECT_ROOT / ".env"


class ProjectConfig(BaseModel):
    # artifacts_destination: str
    tracking_server_uri: str
    # artifacts_datasets_dir: str
    logging_precision: int
    train_dataset_name: str
    test_dataset_name: str
    datasets_file_format: str
    dvc_remote_name: str
    local_datasets_dir: str


class StorageConfig(BaseModel):
    url: str
    endpointurl: str
    region: str
    access_key_id: str
    secret_access_key: str


class ModelConfig(BaseModel):
    save_dir: str
    name: str
    type_: str
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
    storage: Optional[StorageConfig]
    project: ProjectConfig
    model: ModelConfig


def find_config_file(cfg_path: Path) -> Path:
    if cfg_path.is_file():
        return cfg_path
    raise FileNotFoundError(f"Config not found at {cfg_path}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file(CONFIG_FILE_PATH)

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config


def fetch_config_from_dotenv(
        cfg_path: Optional[Path] = None) -> Optional[Dict[str, str]]:
    if not cfg_path:
        try:
            cfg_path = find_config_file(STORAGE_CONFIG_FILE_PATH)
        except FileNotFoundError:
            return None

    if cfg_path:
        env_vars = dotenv_values(cfg_path)
        parsed_config = {
            "url": env_vars["DVC_REMOTE_URL"],
            "endpointurl": env_vars["AWS_ENDPOINT_URL"],
            "region": env_vars["AWS_DEFAULT_REGION"],
            "access_key_id": env_vars["AWS_ACCESS_KEY_ID"],
            "secret_access_key": env_vars["AWS_SECRET_ACCESS_KEY"]
        }
        return parsed_config


def create_and_validate_config(
        parsed_config: Optional[YAML] = None,
        parsed_storage_config: Optional[Dict[str, str]] = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()
    if parsed_storage_config is None:
        parsed_storage_config = fetch_config_from_dotenv()

    _config = Config(
        storage=StorageConfig(**parsed_storage_config)
        if parsed_storage_config is not None else None,
        project=ProjectConfig(**parsed_config.data),
        model=ModelConfig(**parsed_config.data)
    )

    return _config


config = create_and_validate_config()
