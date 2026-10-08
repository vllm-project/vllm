# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from vllm.config import VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.outputs import LateChunk, LateChunkingMetadata, PoolingRequestOutput
from vllm.renderers import TokenizeParams

from .typing import AnyRenderParam, EncodeCMPLRenderParams


def prepare_late_chunking_input(
    config: VllmConfig, render_params: AnyRenderParam
) -> tuple[str, TokenizeParams]:
    """Validate full-document text input before requesting tokenizer offsets."""
    params = render_params["params"]
    params.verify(config.model_config)
    if (
        config.cache_config.enable_prefix_caching
        or config.scheduler_config.enable_chunked_prefill
        or config.lora_config is not None
        or render_params["lora_requests"] is not None
    ):
        raise VLLMValidationError(
            "Late chunking does not support prefix caching, chunked prefill or LoRA"
        )
    if "prompts" not in render_params:
        raise VLLMValidationError("Late chunking requires plain-text input")
    prompt = cast(EncodeCMPLRenderParams, render_params)["prompts"]
    if (
        not isinstance(prompt, dict)
        or set(prompt) - {"prompt", "cache_salt"}
        or not isinstance(text := prompt.get("prompt"), str)
        or not text
    ):
        raise VLLMValidationError("Late chunking requires nonempty plain-text input")
    tok_params = render_params["tok_params"]
    if (
        tok_params.pad_prompt_tokens is not None
        or tok_params.truncate_prompt_tokens is not None
        or tok_params.do_lower_case
    ):
        raise VLLMValidationError(
            "Late chunking does not support input padding, truncation or "
            "renderer text normalization"
        )
    return text, replace(tok_params, return_token_offsets=True)


def build_late_chunking_metadata(
    text: str,
    num_tokens: int,
    offsets: Sequence[tuple[int, int]] | None,
    chunk_size: int,
) -> LateChunkingMetadata:
    """Map chunks to original text, keeping special-token-only chunks."""
    if num_tokens <= 0 or offsets is None or len(offsets) != num_tokens:
        raise VLLMValidationError(
            "Late chunking requires a fast tokenizer with offsets aligned to "
            "all input tokens"
        )
    previous_start = previous_end = 0
    for start, end in offsets:
        if not 0 <= start <= end <= len(text):
            raise VLLMValidationError("Token offsets are not aligned to the input text")
        if start < end:
            if start < previous_start or end < previous_end:
                raise VLLMValidationError("Token offsets are not in source-text order")
            previous_start, previous_end = start, end

    chunks = []
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        source = [(a, b) for a, b in offsets[start:end] if a < b]
        char_range = (
            (min(a for a, _ in source), max(b for _, b in source)) if source else None
        )
        chunks.append(LateChunk(token_range=(start, end), char_range=char_range))
    return LateChunkingMetadata(
        chunk_size=chunk_size, input_tokens=num_tokens, chunks=chunks
    )


def attach_late_chunking_metadata(
    output: PoolingRequestOutput, metadata: LateChunkingMetadata
) -> None:
    """Attach ranges only to successful, complete chunk-vector results."""
    if not output.finished or output.error is not None:
        return
    data = output.outputs.data
    if (
        data.ndim != 2
        or data.shape[0] != len(metadata.chunks)
        or len(output.prompt_token_ids) != metadata.input_tokens
    ):
        raise ValueError("Late-chunking output does not match its token/source ranges")
    output.late_chunking = metadata
