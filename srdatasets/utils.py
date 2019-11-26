import json
import logging
import os
from pathlib import Path

from srdatasets.datasets import __datasets__, dataset_classes

logger = logging.getLogger(__name__)

__warehouse__ = Path(os.path.expanduser("~")).joinpath(".srdatasets")


def get_processed_datasets():
    P = [
        "dev/train.pkl",
        "dev/test.pkl",
        "dev/stats.json",
        "test/train.pkl",
        "test/test.pkl",
        "test/stats.json",
        "config.json",
    ]
    D = {}
    for d in __warehouse__.iterdir():  # loop datasets
        if d.joinpath("processed").exists():
            configs = []
            for c in d.joinpath("processed").iterdir():  # loop configs
                if all(c.joinpath(p).exists() for p in P):
                    configs.append(c.stem)
            if configs:
                D[d.stem] = configs
    return D


def get_downloaded_datasets():
    """ Simple check based on the existences of corefiles 
    """
    D = []
    for d in __datasets__:
        if "-" in d:
            corefile = dataset_classes[d.split("-")[0]].__corefile__[d.split("-")[1]]
        else:
            corefile = dataset_classes[d].__corefile__
        if isinstance(corefile, list):
            if all(__warehouse__.joinpath(d, "raw", cf).exists() for cf in corefile):
                D.append(d)
        else:
            if __warehouse__.joinpath(d, "raw", corefile).exists():
                D.append(d)
    return D


def read_json(path):
    content = {}
    if path.exists():
        with open(path, "r") as f:
            try:
                content = json.load(f)
            except:
                logger.exception("Read json file failed")
    return content


def get_datasetname(name):
    return {d.lower(): d for d in __datasets__}.get(name.lower(), name)
