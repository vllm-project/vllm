# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List

# Third Party
from transformers import AutoConfig

# First Party
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import QuantizationSpec

logger = init_logger(__name__)


@dataclass
class CacheGenConfig:
    # TODO: move this class to another file like "cachegen_basics.py"
    nlayers: int
    kspecs: List[QuantizationSpec]
    vspecs: List[QuantizationSpec]

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        family_7b = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "lmsys/longchat-7b-16k",
            "Qwen/Qwen-7B",
        ]
        family_8b = ["meta-llama/Llama-3.1-8B-Instruct"]
        family_9b = ["THUDM/glm-4-9b-chat"]
        if model_name in family_7b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        elif model_name in family_8b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        # TODO(Jiayi): needs tuning for better quality
        elif model_name in family_9b:
            return CacheGenConfig(
                nlayers=40,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=40, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=40, bins=16),
                ],
            )
        elif model_name == "test_model":
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        else:
            try:
                config = AutoConfig.from_pretrained(model_name)
                # Default name caught by num_hidden_layers
                if config.num_hidden_layers is None:
                    raise ValueError(
                        f"num_hidden_layers is None for model {model_name}"
                    )
                if config.num_hidden_layers < 10:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32,
                            ),
                        ],
                        vspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32,
                            ),
                        ],
                    )
                else:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                            QuantizationSpec(
                                start_layer=10,
                                end_layer=config.num_hidden_layers,
                                bins=16,
                            ),
                        ],
                        vspecs=[
                            QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                            QuantizationSpec(
                                start_layer=2,
                                end_layer=config.num_hidden_layers,
                                bins=16,
                            ),
                        ],
                    )
            except Exception as e:
                raise ValueError(
                    f"Model {model_name} not supported by CacheGenConfig"
                ) from e
