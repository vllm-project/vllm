from typing import Optional

from collections.abc import Mapping
from dataclasses import dataclass, fields
import torch


@dataclass
class KVCache(Mapping):
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None

    # make KVCache a Mapping
    def __iter__(self):
        for field in fields(self):
            if getattr(self, field.name) is None:
                continue
            yield field.name

    def __len__(self):
        num_items = 0
        for item in iter(self):
            num_items += int(item is not None)
        return num_items

    def __getitem__(self, name):
        return getattr(self, name)

    def zero_(self):
        """Set all key and value_cache buffers to zero"""
        if self.key_cache is not None:
            self.key_cache.zero_()
        if self.value_cache is not None:
            self.value_cache.zero_()
