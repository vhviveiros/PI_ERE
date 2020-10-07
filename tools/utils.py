import os
from shutil import rmtree


def check_folder(folder, remove_previous=True):
    if os.path.exists(folder):
        if remove_previous:
            rmtree(folder)
            os.makedirs(folder)
    else:
        os.makedirs(folder)


def path_join(path, *paths):
    return os.path.join(path, *paths)


def abs_path(path, *paths):
    return path_join(os.path.abspath(path), *paths)


def real_path(file, *paths):
    return path_join(os.path.dirname(os.path.realpath(file)), *paths)
