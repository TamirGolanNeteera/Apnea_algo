import json
import os

import numpy as np
import pandas as pd

from Tests.Utils.PathUtils import create_dir


def load(path: str):
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    else:
        return np.load(path.replace(' ', ''), allow_pickle=True)


def save(obj, path: str):
    create_dir(os.path.dirname(path))
    print(f'saving: {path}')
    if path.endswith('json'):
        with open(path, 'w') as f:
            json.dump(obj, f)
    elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        obj.to_pickle(path)
    else:
        np.save(path, obj)
