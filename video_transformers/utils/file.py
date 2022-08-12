import glob
import os
import re
import urllib.request
from pathlib import Path


def increment_path(path, exist_ok=True, sep=""):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def download_file(url: str, download_path: str):
    """
    Downloads a file from the given url to the given path.
    """
    if not os.path.exists(download_path):
        print(f"Downloading {url} to {download_path}")
        urllib.request.urlretrieve(url, download_path)
    else:
        print(f"{download_path} already exists. Skipping download.")
