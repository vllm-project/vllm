# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Sequence

import torch

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)

# With `--enable-log-request-prompts`, INFO logs may include a bounded prompt
# preview when `--max-log-len` is unset. Full inputs remain at DEBUG.
# See github.com/vllm-project/vllm/issues/38537.
_DEFAULT_INFO_PROMPT_STR_LEN = 4096
_DEFAULT_INFO_PROMPT_TOKEN_IDS = 512


class RequestLogger:
    def __init__(
        self,
        *,
        max_log_len: int | None,
        log_prompts_at_info: bool = False,
    ) -> None:
        self.max_log_len = max_log_len
        self.log_prompts_at_info = log_prompts_at_info

        if not logger.isEnabledFor(logging.INFO):
            logger.warning_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than INFO. "
                "No request information will be logged."
            )
        elif self.log_prompts_at_info and not logger.isEnabledFor(logging.DEBUG):
            logger.info_once(
                "`--enable-log-request-prompts` is set but "
                "the minimum log level is higher than DEBUG. "
                "Prompt details at INFO are truncated when long; "
                "set `VLLM_LOGGING_LEVEL=DEBUG` for full details."
            )
        elif not logger.isEnabledFor(logging.DEBUG):
            logger.info_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than DEBUG. "
                "Only limited information will be logged to minimize overhead. "
                "To view more details, set `VLLM_LOGGING_LEVEL=DEBUG`."
            )

    def _prompt_summary_for_info(
        self,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
    ) -> str:
        if not self.log_prompts_at_info or not logger.isEnabledFor(logging.INFO):
            return ""

        max_chars = (
            self.max_log_len
            if self.max_log_len is not None
            else _DEFAULT_INFO_PROMPT_STR_LEN
        )
        max_ids = (
            self.max_log_len
            if self.max_log_len is not None
            else _DEFAULT_INFO_PROMPT_TOKEN_IDS
        )
        if prompt is not None:
            preview = prompt[:max_chars]
            return f", prompt: {preview!r}"
        if prompt_token_ids is not None:
            preview_ids = prompt_token_ids[:max_ids]
            return f", prompt_token_ids: {preview_ids}"
        if prompt_embeds is not None:
            return f", prompt_embeds: shape={prompt_embeds.shape}"
        return ""

    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            max_log_len = self.max_log_len
            if max_log_len is not None:
                if prompt is not None:
                    prompt = prompt[:max_log_len]

                if prompt_token_ids is not None:
                    prompt_token_ids = prompt_token_ids[:max_log_len]

            logger.debug(
                "Request %s details: prompt: %r, "
                "prompt_token_ids: %s, "
                "prompt_embeds shape: %s.",
                request_id,
                prompt,
                prompt_token_ids,
                prompt_embeds.shape if prompt_embeds is not None else None,
            )

        prompt_summary = self._prompt_summary_for_info(
            prompt, prompt_token_ids, prompt_embeds
        )
        logger.info(
            "Received request %s: params: %s, lora_request: %s%s.",
            request_id,
            params,
            lora_request,
            prompt_summary,
        )

    def log_outputs(
        self,
        request_id: str,
        outputs: str,
        output_token_ids: Sequence[int] | None,
        finish_reason: str | None = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None:
        max_log_len = self.max_log_len
        if max_log_len is not None:
            if outputs is not None:
                outputs = outputs[:max_log_len]

            if output_token_ids is not None:
                # Convert to list and apply truncation
                output_token_ids = list(output_token_ids)[:max_log_len]

        stream_info = ""
        if is_streaming:
            stream_info = " (streaming delta)" if delta else " (streaming complete)"

        logger.info(
            "Generated response %s%s: output: %r, "
            "output_token_ids: %s, finish_reason: %s",
            request_id,
            stream_info,
            outputs,
            output_token_ids,
            finish_reason,
        )
