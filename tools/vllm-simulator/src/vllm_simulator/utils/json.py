import json
from array import array
from dataclasses import asdict, is_dataclass
from enum import Enum

import numpy as np


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Enum
        if isinstance(obj, Enum):
            return obj.value
        # Dataclass
        if is_dataclass(obj):
            return asdict(obj)
        # Numpy
        if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
            return int(obj) if isinstance(obj, (np.int32, np.int64)) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Array
        if isinstance(obj, array):
            return obj.tolist()
        # Other
        return super().default(obj)