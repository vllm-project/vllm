# Adapted from
#  https://github.com/modelscope/ms-swift/blob/v2.4.2/swift/utils/module_mapping.py

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Union


class ModelComposeMethod(IntEnum):
    """
    `ModelComposeMethod` distinguishes between two architectural patterns in 
    multi-modal models, focusing on how vision model, language model, and 
    projector are implemented:
    1. Decoupled Architecture (like mllama, InternVL, miniCPMV), complete 
    independent implementation with its own layers, for example:
    ```
    InternVLChatModel
    ├── vision_model (visual encoder)
    │   ├── embeddings
    │   └── encoder
    ├── language_model (language model)
    │   ├── tok_embeddings
    │   └── layers
    └── mlp1 (projector)
    ```
    2. Coupled Architecture (like QWenVL, GLM4V), Integrated as a sub-module 
    with shared architectural patterns , for example: 
    
    ```
    QWenVL
    └── transformer
        ├── wte
        ├── h (language model)
        ├── ln_f
        └── visual (visual encoder)
            ├── conv1
            ├── transformer
            └── attn_pool (projector)
    ```
    """
    Decoupled = 0
    Coupled = 1


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

    compose_type: str = None


@dataclass
class MultiModelKeys(ModelKeys):
    language_model: List[str] = field(default_factory=list)
    connector: List[str] = field(default_factory=list)
    # vision tower and audio tower
    tower_model: List[str] = field(default_factory=list)
    generator: List[str] = field(default_factory=list)

    @staticmethod
    def from_string_field(language_model: Union[str, List[str]] = None,
                          connector: Union[str, List[str]] = None,
                          tower_model: Union[str, List[str]] = None,
                          generator: Union[str, List[str]] = None,
                          compose_type: str = None,
                          **kwargs) -> 'MultiModelKeys':
        assert compose_type, "compose_type is not allowed to be None"

        def to_list(value):
            if value is None:
                return []
            return [value] if isinstance(value, str) else list(value)

        return MultiModelKeys(language_model=to_list(language_model),
                              connector=to_list(connector),
                              tower_model=to_list(tower_model),
                              generator=to_list(generator),
                              compose_type=compose_type,
                              **kwargs)
