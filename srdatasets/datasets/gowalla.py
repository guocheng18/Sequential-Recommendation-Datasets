import glob
import gzip
import logging
import os
import shutil
from datetime import datetime

import pandas as pd
import wget

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class Gowalla(Dataset):

    __url__ = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"
    __corefile__ = "loc-gowalla_totalCheckins.txt"

    def download(self) -> None:
        try:
            wget.download(self.__url__, out=self.home.as_posix())
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(
                os.path.join(os.getcwd(), "loc-gowalla_totalCheckins.txt.gz*.tmp")
            ):
                os.remove(f)
            return

        zipfile_path = self.home.joinpath("loc-gowalla_totalCheckins.txt.gz")
        unzipfile_path = self.home.joinpath("loc-gowalla_totalCheckins.txt")

        with gzip.open(zipfile_path, "rb") as f_in:
            with open(unzipfile_path, "w") as f_out:
                shutil.copyfileobj(f_in, f_out)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self) -> pd.DataFrame:
        """ Time: yyyy-mm-ddThh:mm:ssZ -> timestamp """
        df = pd.read_csv(
            self.home.joinpath(self.__corefile__),
            sep="\t",
            names=["user_id", "timestamp", "latitude", "longtitude", "item_id"],
            index_col=False,
            usecols=[0, 1, 4],
            converters={
                "timestamp": lambda x: int(
                    datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").timestamp()
                )
            },
        )
        return df
