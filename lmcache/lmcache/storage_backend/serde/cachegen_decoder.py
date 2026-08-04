# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenConfig,
    CacheGenGPUBytestream,
    CacheGenGPUEncoderOutput,
)
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.c_ops as lmc_ops
import lmcache.storage_backend.serde.cachegen_basics as CGBasics

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def quant(bins: int, xq: torch.Tensor, max1: float):
    C = bins // 2 - 1
    x = xq / C * max1
    return x


def do_dequantize(t: torch.Tensor, bins: torch.Tensor, maxtensors: torch.Tensor):
    """
    t: [nlayers, ntokens, nchannels]
    bins: [nlayers]
    maxtensors: [nlayers, ntokens, 1]
    """
    C = (bins // 2 - 1)[:, None, None]
    t = t - C
    t = t / C
    t = t * maxtensors
    return t


@_lmcache_nvtx_annotate
def recombine_bytes(bytes_tensor, output_lengths) -> torch.Tensor:
    output_buffer_size = CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK
    offsets = output_lengths.flatten().cumsum(0).roll(1).reshape(output_lengths.shape)
    offsets[0][0] = 0
    indexes = torch.arange(output_buffer_size, device=offsets.device).tile(
        (output_lengths.shape[0], output_lengths.shape[1], 1)
    )
    final_indexes = (indexes + offsets[:, :, None]).clamp(max=len(bytes_tensor) - 1)
    return bytes_tensor[final_indexes]


@_lmcache_nvtx_annotate
def decode_chunk(
    cdf: torch.Tensor,
    data_chunk: CacheGenGPUBytestream,
    target_buffer: torch.Tensor,
) -> None:
    """
    Write the decode output in target_buffer
    Expected shape: [nlayers (kv in total), ntokens, nchannels]
    """
    bytes_tensor = data_chunk.bytestream
    length_prefsum = (
        data_chunk.bytestream_lengths.flatten()
        .cumsum(0)
        .reshape(data_chunk.bytestream_lengths.shape)
    )
    lmc_ops.decode_fast_prefsum(cdf, bytes_tensor, length_prefsum, target_buffer)


@_lmcache_nvtx_annotate
def decode_function_gpu(
    cdf: torch.Tensor,
    data_chunks: List[CacheGenGPUBytestream],
    layers_in_key: int,
    chunk_size: int,
    output: torch.Tensor,
):
    # TODO: dtype and shape -- still have 128 and 8
    """
    Given the path to the encoded KV bytestream, decode the KV cache

    Inputs:
        cdf: the cdf tensor, in shape [2 * nlayers, nchannels, bins + 1]
        data_chunks: the data_chunks in the encoder's output
        layers_in_key: number of layers in K (or V)
        (K/V should have the same number of layers)
        chunk_size: the chunk_size
        output: output buffer, in shape [ntokens, 2 * nlayers * nchannels]

    Outputs:
        key: the decoded key tensor in the shape of (layers, tokens, nchannels)
        value: the decoded value tensor in the shape of
        (layers, tokens, nchannels)
    """
    nlayers, nchannels, _ = cdf.shape
    output = output.reshape((nlayers, chunk_size, nchannels))

    start = 0
    for data_chunk in data_chunks:
        end = start + data_chunk.ntokens
        decode_chunk(cdf, data_chunk, output[:, start:end, :])
        start = end

    out = output.reshape((2, layers_in_key, chunk_size, nchannels))
    key, value = out.float()

    return key, value


class CacheGenDeserializer(Deserializer):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        dtype,
    ):
        self.dtype = dtype
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

    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        encoder_output = CacheGenGPUEncoderOutput.from_bytes(bs)
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
            self.value_bins = self.value_bins.to(value.device)

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

        return blob.permute((1, 0, 2, 3, 4)).to(
            self.dtype
        )  # [nlayers, 2, ntokens, num_heads, head_size]
        # huggingface
        # return blob.permute((1, 0, 3, 2, 4)).to(
        #     self.dtype
        # )  # [nlayers, 2, num_heads, ntokens, head_size]
