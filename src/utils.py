"""
Module which stores helper functions used during model training

Author: Jared Andrews
Date: 6/11/23
"""

import os
import glob
import pickle
from src.config import raw_data_fp


def clean_folder(folder_path):
    """
    Helper functions which clears provided folder

    :param folder_path:
    :return:
        None
    """
    # Iterate over each file in provided folder
    for f in glob.glob(os.path.join(folder_path, '*')):
        # If file isn't the raw data used in training, remove file
        if f != raw_data_fp:
            os.remove(f)


def bool_func(x):
    """
    Helper function used to enable argparse to parse boolean values

    :param x: provided parameter for boolean parameter
    :return: Desired boolean
    """
    return x == "True"


def save_model(model, model_fp):
    """
    Helper function to save trained model files
    input:
            model: fit model to be saved
            model_fp: file path where model is saved to
    output:
            None
    """
    with open(model_fp, 'wb') as file_path:
        pickle.dump(model, file_path)
