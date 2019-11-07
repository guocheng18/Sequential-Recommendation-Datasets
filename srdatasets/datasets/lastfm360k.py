import glob
import logging
import os
import shutil
import tarfile
from urllib.parse import urlparse

import pandas as pd
import wget
from pandas import DataFrame

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class Lastfm360K(Dataset):

    __download_url__ = (
        "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"
    )
    __corefile__ = "usersha1-artmbid-artname-plays.tsv"

    def download(self) -> None:
        try:
            wget.download(self.__download_url__, out=self.home)
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(self.home.joinpath("*.tmp")):
                os.remove(f)
            return

        zipfile_name = os.path.basename(urlparse(self.__download_url__).path)
        zipfile_path = self.home.joinpath(zipfile_name)

        with tarfile.open(zipfile_path) as tar:
            tar.extractall(self.home)

        unzip_folder = self.home.joinpath("lastfm-dataset-360K")
        shutil.move(unzip_folder.joinpath("*"), self.home)
        os.rmdir(unzip_folder)
        logger.info("Finished")

    def transform(self, *args) -> DataFrame:
        pass


# TODO: transform and test
