import gzip
import os
import shutil
import tarfile
from zipfile import ZipFile

import wget

from srdatasets.utils import __storage__


def _download(dataset_name: str):
    _store_dir = os.path.join(__storage__, dataset_name, "raw")

    # Create folder
    if os.path.exists(_store_dir):
        os.remove(os.path.join(_store_dir, "*"))
    else:
        os.makedirs(_store_dir)

    if dataset_name == "movielens-20m":
        wget.download(
            "http://files.grouplens.org/datasets/movielens/ml-20m.zip", out=_store_dir
        )

        _zipped_file = os.path.join(_store_dir, "ml-20m.zip")
        _unzipped_folder = os.path.join(_store_dir, "ml-20m")

        with ZipFile(_zipped_file) as zipObj:
            zipObj.extractall(_store_dir)

        shutil.move(os.path.join(_unzipped_folder, "*"), _store_dir)
        os.rmdir(_unzipped_folder)
        os.remove(_zipped_file)

    elif dataset_name == "lastfm-360k":
        wget.download(
            "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz",
            out=_store_dir,
        )

        _zipped_file = os.path.join(_store_dir, "lastfm-dataset-360K.tar.gz")
        _unzipped_folder = os.path.join(_store_dir, "lastfm-dataset-360K")

        with tarfile.open(_zipped_file) as tar:
            tar.extractall(_store_dir)

        shutil.move(os.path.join(_unzipped_folder, "*"), _store_dir)
        os.rmdir(_unzipped_folder)
        os.remove(_zipped_file)

    elif dataset_name == "gowalla":
        wget.download(
            "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
            out=_store_dir,
        )

        _zipped_file = os.path.join(_store_dir, "loc-gowalla_totalCheckins.txt.gz")
        _unzipped_file = os.path.join(_store_dir, "loc-gowalla_totalCheckins.txt")

        with gzip.open(_zipped_file, "rb") as f_in:
            with open(_unzipped_file, "w") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(_zipped_file)

    print("done.")
