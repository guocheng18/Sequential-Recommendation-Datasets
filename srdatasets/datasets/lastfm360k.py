import os
import shutil
import tarfile

import pandas as pd
import wget
from pandas import DataFrame

from srdatasets.datasets.base import Dataset


class Lastfm360K(Dataset):

    __url__ = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz"

    def download(self) -> None:
        wget.download(self.__url__, out=self.home)

        _zipped_file = self.home.joinpath("lastfm-dataset-360K.tar.gz")
        _unzipped_folder = self.home.joinpath("lastfm-dataset-360K")

        with tarfile.open(_zipped_file) as tar:
            tar.extractall(self.home)

        shutil.move(_unzipped_folder.joinpath("*"), self.home)
        os.rmdir(_unzipped_folder)
        os.remove(_zipped_file)

    def transform(self, *args) -> DataFrame:
        pass

# TODO: transform and test
