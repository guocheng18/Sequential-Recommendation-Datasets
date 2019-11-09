import glob
import logging
import os
import tarfile
from datetime import datetime

import pandas as pd

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class Yelp(Dataset):

    __corefile__ = os.path.join("yelp_dataset", "review.json")

    def download(self) -> None:
        if not self.home.joinpath("yelp_dataset.tar.gz").exists():
            logger.warning(
                "Since Yelp dataset is not directly accessible, please visit https://www.yelp.com/dataset/download  \
                and download it manually, after downloaded, place file 'yelp_dataset.tar.gz' under {} and run this command again".format(
                    self.home
                )
            )
        else:
            with tarfile.open(self.home.joinpath("yelp_dataset.tar.gz")) as tar:
                tar.extractall(self.home)
            logger.info("Dataset ready")

    def transform(self) -> pd.DataFrame:
        pass
