import bz2
import gzip
import logging
import os
import shutil
import tarfile
import urllib.request
from zipfile import ZipFile

from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """ From https://stackoverflow.com/a/53877507/8810037
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        logger.info("Download successful")
    except:
        logger.exception("Download failed, please try again")
        if output_path.exists():
            os.remove(output_path)


def extract(filepath, out):
    """ out: a file or a directory
    """
    logger.info("Unzipping...")
    filename = filepath.as_posix()

    if filename.endswith(".zip") or filename.endswith(".ZIP"):
        with ZipFile(filepath) as zipObj:
            zipObj.extractall(out)

    elif filename.endswith(".tar.gz"):
        with tarfile.open(filepath, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, out)

    elif filename.endswith(".tar.bz2"):
        with tarfile.open(filepath, "r:bz2") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, out)

    elif filename.endswith(".gz"):
        with gzip.open(filepath, "rb") as fin:
            with open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)

    elif filename.endswith(".bz2"):
        with bz2.open(filepath, "rb") as fin:
            with open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    else:
        logger.error("Unrecognized compressing format of %s", filepath)
        return

    logger.info("OK")
