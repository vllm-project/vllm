# Copied code from
#  https://github.com/modelscope/ms-swift/blob/v2.4.2/swift/utils/module_mapping.py

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class ModelKeys:
    model_type: str = None

    module_list: str = None

    embedding: str = None

    mlp: str = None

    down_proj: str = None

    attention: str = None

    o_proj: str = None

    q_proj: str = None

    k_proj: str = None

    v_proj: str = None

    qkv_proj: str = None

    qk_proj: str = None

    qa_proj: str = None

    qb_proj: str = None

    kva_proj: str = None

    kvb_proj: str = None

    output: str = None


@dataclass
class MultiModelKeys(ModelKeys):
    language_model: Union[List[str], str] = field(default_factory=list)
    connector: Union[List[str], str] = field(default_factory=list)
    # vision tower and audio tower
    tower_model: Union[List[str], str] = field(default_factory=list)
    generator: Union[List[str], str] = field(default_factory=list)

    def __post_init__(self):
        for key in ["language_model", "connector", "tower_model", "generator"]:
            v = getattr(self, key)
            if isinstance(v, str):
                setattr(self, key, [v])
            if v is None:
                setattr(self, key, [])
