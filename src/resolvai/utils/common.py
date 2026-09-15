import os
import json
import pickle
import yaml
from pathlib import Path
from typing import Any, List

from resolvai import logger


class AttrDict(dict):
    """Dict that also supports attribute access, e.g. cfg.data_ingestion.raw_data_path."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return AttrDict(value) if isinstance(value, dict) else value


def read_yaml(path: Path) -> AttrDict:
    with open(path) as f:
        content = yaml.safe_load(f) or {}
    logger.info(f"yaml file loaded: {path}")
    return AttrDict(content)


def create_directories(paths: List[Path], verbose: bool = True) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {p}")


def save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"json saved at: {path}")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_bin(data: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"binary saved at: {path}")


def load_bin(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
