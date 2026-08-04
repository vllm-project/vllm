# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List
import io
import pickle

# Third Party
from transformers import AutoConfig
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK = 256


@dataclass
class QuantizationSpec:
    start_layer: int
    end_layer: int
    bins: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


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


@dataclass
class CacheGenEncoderOutput:
    # TODO: maybe use numpy array so that we can directly tobytes() and
    # frombuffer() to have a better performance
    bytestream: bytes
    start_indices: torch.Tensor
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def to_bytes(self) -> bytes:
        """Save the output to a file"""
        with io.BytesIO() as f:
            # torch.save(self, f)
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    def from_bytes(bs: bytes) -> "CacheGenEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)


@dataclass
class CacheGenGPUBytestream:
    bytestream: torch.Tensor
    bytestream_lengths: torch.Tensor  # [nlayers, nchannels, bytestream_length]
    ntokens: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenGPUEncoderOutput:
    data_chunks: List[CacheGenGPUBytestream]
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @_lmcache_nvtx_annotate
    def to_bytes(self) -> bytes:
        """Save the output to a file"""
        with io.BytesIO() as f:
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    @_lmcache_nvtx_annotate
    def from_bytes(bs: bytes) -> "CacheGenGPUEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)

    def debug_print_device(self):
        logger.debug(f"bytestream device: {self.data_chunks[0].bytestream.device}")
        logger.debug(
            f"bytestream_lengths device: "
            f"{self.data_chunks[0].bytestream_lengths.device}"
        )
        logger.debug(f"cdf device: {self.cdf.device}")
        logger.debug(f"max_tensors_key device: {self.max_tensors_key.device}")
        logger.debug(f"max_tensors_value device: {self.max_tensors_value.device}")
