# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenGPUEncoderOutput,
)
from lmcache.storage_backend.serde.cachegen_decoder import (
    decode_function_gpu,
    do_dequantize,
)
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.naive_serde.cachegen_basics import CacheGenConfig
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer

logger = init_logger(__name__)


class CacheGenDeserializer(Deserializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheMetadata):
        self.dtype = metadata.kv_dtype
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.output_buffer: Optional[torch.Tensor] = None
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.kspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.to(torch_device_type)

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.vspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.to(torch_device_type)

    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int):
        if (
            self.output_buffer is None
            or self.output_buffer.shape[1] != 2 * nlayers * nchannels
        ):
            self.output_buffer = torch.zeros(
                (self.chunk_size, 2 * nlayers * nchannels), dtype=torch.uint8
            ).to(torch_device_type)
        return self.output_buffer[:ntokens, :]

    # TODO(Jiayi): A lot of memory copies can be avoided in this function.
    @_lmcache_nvtx_annotate
    def deserialize(self, buffer_memory_obj: BytesBufferMemoryObj) -> MemoryObj:
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(
            buffer_memory_obj.byte_array
        )

        encoder_output.max_tensors_key = encoder_output.max_tensors_key.to(
            torch_device_type
        )
        encoder_output.max_tensors_value = encoder_output.max_tensors_value.to(
            torch_device_type
        )

        ntokens = encoder_output.max_tensors_key.shape[1]
        layers_in_key = encoder_output.max_tensors_key.shape[0]
        key, value = decode_function_gpu(
            encoder_output.cdf,
            encoder_output.data_chunks,
            layers_in_key,
            ntokens,
            self.get_output_buffer(
                encoder_output.cdf.shape[0] // 2,
                encoder_output.cdf.shape[1],
                ntokens,
            ),
        )

        # Temporary fix for #83: change the device of key_bins and value_bins
        # to the device of key and value
        # This requires a long-term fix in the future. Currently,
        # CacheGenGPUEncoderOutput has implicit device in itself.
        # More specifically, if the encoder encodes the tensor on GPU0, the
        # from_bytes will also return a tensor on GPU0
        # We may want to dynamically configure the device based on config and
        # metadata in the future
        if self.key_bins.device != key.device:
            self.key_bins = self.key_bins.to(key.device)

        if self.value_bins.device != value.device:
            self.value_bins = self.value_bins.to(torch_device_type)

        key = do_dequantize(key, self.key_bins, encoder_output.max_tensors_key)
        value = do_dequantize(value, self.value_bins, encoder_output.max_tensors_value)
        """ merge key and value back and reshape """
        nlayers, ntokens, nchannels = key.shape
        blob = torch.stack([key, value])  # [2, nlayers, ntokens, nchannels]
        blob = blob.reshape(
            (
                2,
                nlayers,
                ntokens,
                encoder_output.num_heads,
                encoder_output.head_size,
            )
        )

        hidden_dim = blob.shape[-1] * blob.shape[-2]
        kv_chunk = blob.reshape(*blob.shape[:-2], hidden_dim).to(
            self.dtype
        )  # [nlayers, 2, ntokens, num_heads, head_size]

        memory_obj = TensorMemoryObj(
            raw_data=kv_chunk,
            metadata=MemoryObjMetadata(
                shape=kv_chunk.shape,
                dtype=kv_chunk.dtype,
                address=-1,
                phy_size=kv_chunk.numel() * kv_chunk.element_size(),
                ref_count=-1,  # HACK: avoid mis-free
                fmt=MemoryFormat.KV_2LTD,
            ),
            parent_allocator=None,
        )

        return memory_obj
