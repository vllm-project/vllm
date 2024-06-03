from typing import Dict, NamedTuple, Tuple, Union

import torch


class ModelInputSpec(NamedTuple):
    shape: Tuple[int, ...]
    dtype: Union[str, torch.dtype, None] = None


ExtraModelInputsSpec = Dict[str, ModelInputSpec]