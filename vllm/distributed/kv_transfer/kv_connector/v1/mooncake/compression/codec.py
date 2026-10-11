# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class EncodedChunk:
    array: object
    event: torch.cuda.Event
    u8_bytes: int
    payload_bytes: int | None = None
    raw_tail_bytes: int = 0
    raw_tail_addr: int = 0

    @property
    def codec_bytes(self) -> int:
        if self.payload_bytes is not None:
            return self.payload_bytes
        return int(self.array.buffer_size)  # type: ignore[attr-defined]


class ANSCodec:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        u8_sizes: tuple[int, ...],
    ):
        from nvidia import nvcomp

        self._nvcomp = nvcomp
        self.stream = stream
        self.codec = nvcomp.Codec(algorithm="ANS", cuda_stream=stream.cuda_stream)
        self._compression_configs = {
            size: self.codec.compression_config(size) for size in set(u8_sizes)
        }
        self._decompression_configs = {
            size: self.codec.decompression_config(config)
            for size, config in self._compression_configs.items()
        }
        logger.info(
            "Mooncake ANS codec initialized: u8_sizes=%s stream=%s",
            sorted(self._compression_configs),
            stream.cuda_stream,
        )

    def encode(
        self,
        source: torch.Tensor,
        output: torch.Tensor,
        u8_bytes: int,
        event: torch.cuda.Event,
    ) -> EncodedChunk:
        source_array = self._nvcomp.as_array(
            source[:u8_bytes], cuda_stream=self.stream.cuda_stream
        )
        output_array = self._nvcomp.as_array(
            output, cuda_stream=self.stream.cuda_stream
        )
        encoded = self.codec.encode(
            source_array,
            out=output_array,
            compression_config=self._compression_configs[u8_bytes],
        )
        event.record(self.stream)
        logger.debug(
            "Mooncake ANS encode queued: input_bytes=%d output_capacity=%d ",
            u8_bytes,
            output.numel(),
        )
        return EncodedChunk(encoded, event, u8_bytes)

    def decode(
        self,
        source: torch.Tensor,
        output: torch.Tensor,
        codec_bytes: int,
        u8_bytes: int,
        event: torch.cuda.Event,
    ) -> torch.cuda.Event:
        source_array = self._nvcomp.as_array(
            source[:codec_bytes], cuda_stream=self.stream.cuda_stream
        )
        output_array = self._nvcomp.as_array(
            output[:u8_bytes], cuda_stream=self.stream.cuda_stream
        )
        self.codec.decode(
            source_array,
            out=output_array,
            decompression_config=self._decompression_configs[u8_bytes],
        )
        event.record(self.stream)
        logger.debug(
            "Mooncake ANS decode queued: codec_bytes=%d output_bytes=%d",
            codec_bytes,
            u8_bytes,
        )
        return event
