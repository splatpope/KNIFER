import os

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)