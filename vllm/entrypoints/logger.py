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


class RequestLogger:
    def __init__(self, *, max_log_len: int | None) -> None:
        self.max_log_len = max_log_len
        self.rejected_log_counter = 0
        self.rejected_log_interval = 100
        if not logger.isEnabledFor(logging.INFO):
            logger.warning_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than INFO. "
                "No request information will be logged."
            )
        elif not logger.isEnabledFor(logging.DEBUG):
            logger.info_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than DEBUG. "
                "Only limited information will be logged to minimize overhead. "
                "To view more details, set `VLLM_LOGGING_LEVEL=DEBUG`."
            )

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

        logger.info(
            "Received request %s: params: %s, lora_request: %s.",
            request_id,
            params,
            lora_request,
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

    def log_rejected_request(self, request_id: str) -> None:
        if self.rejected_log_counter == self.rejected_log_interval:
            logger.warning(
                "Request %s was rejected due to "
                "a full waiting queue (log every %d requests)",
                request_id,
                self.rejected_log_interval,
            )
            self.rejected_log_counter = 0
        self.rejected_log_counter += 1
