import glob
import logging
import os
from datetime import datetime

import pandas as pd
import wget

from srdatasets.datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class BookCrossing(Dataset):

    __url__ = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
    __corefile__ = os.path.join("BX-CSV-Dump", "BX-Book-Ratings.csv")

    def download(self) -> None:
        try:
            wget.download(self.__url__, out=self.home.as_posix())
            logger.info("Download successful, unzipping...")
        except:
            logger.exception("Download failed, please retry")
            for f in glob.glob(os.path.join(os.getcwd(), "BX-CSV-Dump.zip*.tmp")):
                os.remove(f)
            return

        with ZipFile(self.home.joinpath("BX-CSV-Dump.zip")) as zipObj:
            zipObj.extractall(self.home)

        logger.info("Finished, dataset location: %s", self.home)

    def transform(self) -> pd.DataFrame:
        pass
