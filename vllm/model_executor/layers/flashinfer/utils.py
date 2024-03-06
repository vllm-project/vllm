"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from enum import Enum


class PosEncodingMode(Enum):
    NONE = 0
    ROPE_LLAMA = 1
    ALIBI = 2


class TensorLayout(Enum):
    NHD = 0
    HND = 1


def expand_5d(x: torch.Tensor, kv_layout: str):
    if not x.ndim in [4, 5]:
        raise ValueError("x must be 4D or 5D")
    if x.ndim == 4:
        # page_size == 1
        if kv_layout == "NHD":
            # expand to 5D on the 3nd last dimension
            return x.unsqueeze(-3)
        elif kv_layout == "HND":
            # expand to 5D on the 2nd last dimension
            return x.unsqueeze(-2)
        else:
            raise KeyError("Invalid kv_layout {}".format(kv_layout))
    return x


def check_pos_encoding_mode(pos_encoding_mode: str):
    if not hasattr(PosEncodingMode, pos_encoding_mode):
        raise KeyError("Invalid pos_encoding_mode {}".format(pos_encoding_mode))


def check_kv_layout(kv_layout: str):
    if not hasattr(TensorLayout, kv_layout):
        raise KeyError("Invalide kv_layout {}".format(kv_layout))


def is_float8(x: torch.Tensor):
    return x.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
