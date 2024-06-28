import numpy as np
from collections import defaultdict
from typing import Dict, List

_POOL: Dict[int, List[np.ndarray]] = defaultdict(list)

def alloc_array(max_tokens: int) -> np.ndarray:
    if max_tokens in _POOL and _POOL[max_tokens]:
        return _POOL[max_tokens].pop()
    return np.zeros((max_tokens, ), dtype=np.int64)

def del_array(arr: np.ndarray) -> None:
    _POOL[len(arr)].append(arr)
