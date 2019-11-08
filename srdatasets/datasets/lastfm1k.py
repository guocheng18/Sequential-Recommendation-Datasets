import glob
import logging
import os
import shutil
import tarfile
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import wget
from pandas import DataFrame

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class Lastfm1K(Dataset):

    __download_url__ = (
        "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
    )
    __corefile__ = "userid-timestamp-artid-artname-traid-traname.tsv"

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

        unzip_folder = self.home.joinpath("lastfm-dataset-1K")
        shutil.move(unzip_folder.joinpath("*"), self.home)
        os.rmdir(unzip_folder)
        logger.info("Finished, dataset location: %s", self.home)

    def transform(self, item_type="song") -> DataFrame:
        """ item_type can be `artist` or `song`
        """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            sep="\t",
            names=[
                "user_id",
                "timestamp",
                "artist_id",
                "artist_name",
                "song_id",
                "song_name",
            ],
            index_col=False,
            usecols=[0, 1, 2, 4],
            converters={
                "timestamp": lambda x: int(
                    datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp()
                )
            },
        )
        if item_type == "song":
            df = df.drop("artist_id").rename(columns={"song_id": "item_id"})
        else:
            df = df.drop("song_id").rename(columns={"artist_id": "item_id"})
        return df
