import pickle
from pathlib import Path


def write_pickle(obj, path):
    """ Write any object to pickle

    Parameters
    ----------
    obj : any object
    path : path-like
        path to storage file.
    """

    path = Path(path)
    # Don't overwrite existing files
    # path.mkdir(parents=True, exist_ok=False)
    raw = pickle.dumps(obj)
    with open(path, "wb") as f:
        raw = f.write(raw)


def read_pickle(path):
    """ Read a stored object
    """
    with open(path, "rb") as f:
        raw = f.read()
    return pickle.loads(raw)
