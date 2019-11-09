import glob
import logging
import os
from datetime import datetime
from zipfile import ZipFile

import pandas as pd
import wget

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class FourSquareTokyo(Dataset):

    __url__ = "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip"
    __corefile__ = os.path.join("dataset_tsmc2014", "dataset_TSMC2014_TKY.txt")

    def download(self) -> None:
        try:
            wget.download(self.__url__, out=self.home.as_posix())
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(os.path.join(os.getcwd(), "dataset_tsmc2014.zip*.tmp")):
                os.remove(f)
            return

        zipfile_path = self.home.joinpath("dataset_tsmc2014.zip")
        unzip_folder = self.home.joinpath("dataset_tsmc2014")

        with ZipFile(zipfile_path) as zipObj:
            zipObj.extractall(self.home)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self) -> pd.DataFrame:
        pass
