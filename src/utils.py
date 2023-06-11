import os
import glob


def clean_folder(folder_path, keep_file='None'):
    for f in glob.glob(os.path.join(folder_path, '*')):
        if keep_file not in f:
            os.remove(f)

