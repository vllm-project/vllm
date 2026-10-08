# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from filelock import FileLock

from vllm.entrypoints.pooling.typing import AnyPoolingRequest
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class RequestLogger:
    def __init__(
        self, *, max_log_len: int | None, log_requests_path: str | None = None
    ) -> None:
        self.max_log_len = max_log_len
        self.log_requests_path = log_requests_path

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
                "To view more details, set `--log-level DEBUG`."
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

    def _write_to_file(self, body: str) -> None:
        if self.log_requests_path is None:
            return
        log_requests_path = Path(self.log_requests_path)
        log_requests_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            FileLock(log_requests_path.with_suffix(".lock")),
            log_requests_path.open("a", encoding="utf-8") as file,
        ):
            file.write(body)
            file.write("\n")

    def log_request_body(self, request: AnyRequest | AnyPoolingRequest) -> None:
        body = request.model_dump(mode="json", exclude_unset=True)
        body["request_type"] = type(request).__name__
        json_body_str = json.dumps(body)
        self._write_to_file(json_body_str)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Request %s JSON body: %s",
                getattr(request, "request_id", "N/A"),
                json_body_str[: self.max_log_len],
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
        if max_log_len is not None and outputs is not None:
            outputs = outputs[:max_log_len]

        stream_info = ""
        if is_streaming:
            stream_info = " (streaming delta)" if delta else " (streaming complete)"

        if logger.isEnabledFor(logging.DEBUG):
            if max_log_len is not None and output_token_ids is not None:
                output_token_ids = list(output_token_ids)[:max_log_len]

            logger.debug(
                "Generated response %s%s details: output_token_ids: %s",
                request_id,
                stream_info,
                output_token_ids,
            )

        logger.info(
            "Generated response %s%s: output: %r, finish_reason: %s",
            request_id,
            stream_info,
            outputs,
            finish_reason,
        )
