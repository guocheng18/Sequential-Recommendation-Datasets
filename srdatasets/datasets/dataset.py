import os
from abc import ABC, abstractmethod
from urllib.parse import urlparse


class Dataset(ABC):
    """ Base dataset of SR datasets
    """

    def __init__(self, rootdir):
        """ `rootdir` is the directory of the raw dataset """
        self.rootdir = rootdir

    @property
    def rawpath(self):
        if hasattr(self, "__url__"):
            return self.rootdir.joinpath(os.path.basename(urlparse(self.__url__).path))
        else:
            return ""

    @abstractmethod
    def download(self):
        """ Download and extract the raw dataset """
        pass

    @abstractmethod
    def transform(self):
        """ Transform to the general data format, which is
        a pd.DataFrame instance that contains three columns: [user_id, item_id, timestamp]
        """
        pass
