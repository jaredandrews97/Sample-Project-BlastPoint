import os
import glob
from src.config import raw_data_fp


def clean_folder(folder_path):
    for f in glob.glob(os.path.join(folder_path, '*')):
        if f != raw_data_fp:
            os.remove(f)


def bool_func(x):
    return x == "True"
