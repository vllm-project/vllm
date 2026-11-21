# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline lm_head inputs for generated positions and explicit prompt scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


@dataclass(frozen=True)
class HiddenStatePlan:
    prompt_len: int
    start: int
    end: int | None
    generated: bool = True
    prompt: bool = False

    @classmethod
    def for_request(cls, request: Request) -> HiddenStatePlan | None:
        params = request.sampling_params
        if params is None or not params.aux_outputs:
            return None
        generated = "hidden_states" in params.aux_outputs
        prompt = "prompt_hidden_states" in params.aux_outputs
        needs_generated = generated and (
            request.num_output_tokens == 0
            or params.hidden_states_end is None
            or request.num_tokens - 1 < params.hidden_states_end
        )
        needs_prompt = (
            prompt and request.num_computed_tokens < request.num_prompt_tokens - 1
        )
        if not (needs_generated or needs_prompt):
            return None
        return cls(
            request.num_prompt_tokens,
            params.hidden_states_start,
            params.hidden_states_end,
            generated,
            prompt,
        )

    def select(self, step_start: int, step_end: int) -> tuple[int, int]:
        start = max(step_start, self.prompt_len - 1, self.start)
        end = step_end if self.end is None else min(step_end, self.end)
        return start, max(start, end)


def verify_hidden_config(config: VllmConfig) -> None:
    if config.device_config.device_type != "cuda":
        raise ValueError("Hidden states require the CUDA Model Runner V2")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Hidden states require async_scheduling=False")
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
        or parallel.enable_batch_sharded_sampling
    ):
        raise ValueError(
            "Hidden states do not support PP, CP or batch-sharded sampling"
        )
    if config.model_config.is_encoder_decoder or config.model_config.is_diffusion:
        raise ValueError("Hidden states require a causal decoder model")
    if (
        config.speculative_config is not None
        and config.speculative_config.enable_adaptive_verification
    ):
        raise ValueError("Hidden states do not support adaptive verification")
    if config.lora_config is not None or config.model_config.quantization is not None:
        raise ValueError(
            "Inline hidden states currently require an unquantized model without LoRA"
        )


def validate_capture_request(
    params: SamplingParams,
    prompt_len: int,
    config: VllmConfig,
    resumable: bool = False,
) -> None:
    if not params.aux_outputs:
        return
    names = set(params.aux_outputs)
    if names - {"hidden_states", "prompt_hidden_states"} or len(names) != len(
        params.aux_outputs
    ):
        raise VLLMValidationError("Unsupported or repeated auxiliary output")
    if not config.aux_output_config.enable_return_hidden_states:
        raise VLLMValidationError("Hidden states require --enable-return-hidden-states")
    start, end = params.hidden_states_start, params.hidden_states_end
    limit = config.model_config.max_model_len
    if (
        type(start) is not int
        or not 0 <= start < limit
        or (end is not None and (type(end) is not int or not start < end <= limit))
    ):
        raise VLLMValidationError(
            "Hidden-state range must satisfy 0 <= start < end <= max_model_len"
        )
    if resumable:
        raise VLLMValidationError("Hidden states do not support streaming input")
    if "prompt_hidden_states" in names and config.cache_config.kv_sharing_fast_prefill:
        raise VLLMValidationError("Prompt hidden states do not support fast prefill")
    if "hidden_states" in names:
        if (
            params.presence_penalty != 0
            or params.frequency_penalty != 0
            or params.repetition_penalty != 1
            or params.logit_bias
        ):
            raise VLLMValidationError(
                "Hidden states do not support penalties or logit_bias"
            )
        masked = (
            params.top_p < 1
            or params.min_p > 0
            or params.allowed_token_ids
            or params.bad_words
            or params.structured_outputs is not None
        )
        if params.top_k > 256 or (masked and params.top_k <= 0):
            raise VLLMValidationError(
                "Truncated or masked hidden-state requests require 1 <= top_k <= 256"
            )
    maximum = min(prompt_len - 1 + (params.max_tokens or limit), limit - 1)
    if end is not None:
        maximum = min(maximum, end)
    rows = (
        max(0, maximum - max(start, prompt_len - 1)) if "hidden_states" in names else 0
    )
    if "prompt_hidden_states" in names:
        rows += max(0, min(end or prompt_len - 1, prompt_len - 1) - start)
    size = torch.empty((), dtype=config.model_config.dtype).element_size()
    nbytes = rows * config.model_config.get_hidden_size() * size
    if nbytes > config.aux_output_config.max_hidden_capture_bytes:
        raise VLLMValidationError(
            "Hidden states exceed max_hidden_capture_bytes per completion"
        )


def hidden_states_metadata(
    model: torch.nn.Module, config: VllmConfig
) -> dict[str, Any]:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    if type(model) is not Qwen3ForCausalLM:
        raise ValueError("Inline hidden states currently support Qwen3ForCausalLM only")
    processor = model.logits_processor
    state_cls = resolve_model_state_cls(config, model)
    if (
        type(processor) is not LogitsProcessor
        or processor.logits_as_input
        or state_cls.compute_sample_logits is not ModelState.compute_sample_logits
    ):
        raise ValueError("Model logits path does not expose the exact lm_head input")
    tied = bool(
        getattr(config.model_config.hf_text_config, "tie_word_embeddings", False)
    )
    return {
        "revision": config.model_config.revision,
        "hidden_size": config.model_config.get_hidden_size(),
        "hidden_dtype": str(config.model_config.dtype).removeprefix("torch."),
        "head_dtype": str(
            processor.head_dtype or config.model_config.dtype
        ).removeprefix("torch."),
        "vocab_size": int(processor.org_vocab_size),
        "logit_scale": float(processor.scale),
        "soft_cap": processor.soft_cap,
        "transform_order": ["projection", "vocab_trim", "soft_cap", "scale"],
        "tie_word_embeddings": tied,
        "lm_head_weight": "model.embed_tokens.weight" if tied else "lm_head.weight",
        "codec": "raw",
    }


@dataclass
class PendingHiddenOutput:
    request_ids: list[str]
    # request id -> (selected absolute start, full step end, owned GPU rows)
    snapshots: dict[str, tuple[int, int, torch.Tensor]]
    prompt_snapshots: dict[str, tuple[int, torch.Tensor]] = field(default_factory=dict)
    prompt_rows: dict[str, torch.Tensor] = field(default_factory=dict)
    rows: dict[str, torch.Tensor] = field(default_factory=dict)
    num_sampled: np.ndarray | None = None
    num_rejected: np.ndarray | None = None

    def enqueue_cpu_copy(
        self, num_sampled: np.ndarray, num_rejected: np.ndarray
    ) -> None:
        self.num_sampled, self.num_rejected = num_sampled, num_rejected
        for request_id, (_, _, rows) in self.snapshots.items():
            cpu = torch.empty_like(rows, device="cpu", pin_memory=True)
            cpu.copy_(rows, non_blocking=True)
            self.rows[request_id] = cpu
        for request_id, (_, rows) in self.prompt_snapshots.items():
            cpu = torch.empty_like(rows, device="cpu", pin_memory=True)
            cpu.copy_(rows, non_blocking=True)
            self.prompt_rows[request_id] = cpu

    def process_prompt_output(self) -> dict[str, tuple[int, torch.Tensor]]:
        return {
            request_id: (start, self.prompt_rows[request_id])
            for request_id, (start, _) in self.prompt_snapshots.items()
        }

    def process_output(self) -> dict[str, tuple[int, torch.Tensor]]:
        assert self.num_sampled is not None and self.num_rejected is not None
        outputs = {}
        for index, request_id in enumerate(self.request_ids):
            if request_id not in self.snapshots or self.num_sampled[index] == 0:
                continue
            start, step_end, _ = self.snapshots[request_id]
            # Rejection is relative to the original logits range, before windowing.
            accepted_end = step_end - int(self.num_rejected[index])
            outputs[request_id] = (
                start,
                self.rows[request_id][: max(0, accepted_end - start)],
            )
        return outputs


class HiddenStateCapturer:
    def __init__(self, model: torch.nn.Module, config: VllmConfig) -> None:
        self.metadata = hidden_states_metadata(model, config)
        logger.info("Inline hidden-state metadata: %s", self.metadata)
        self.plans: dict[str, HiddenStatePlan] = {}
        self.pending: PendingHiddenOutput | None = None

    def prepare(
        self,
        batch: InputBatch,
        sample_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        if not self.plans:
            self.pending = None
            return
        snapshots = {}
        prompt_snapshots = {}
        for index, request_id in enumerate(batch.req_ids):
            plan = self.plans.get(request_id)
            if plan is None:
                continue
            first, last = batch.cu_num_logits_np[index : index + 2]
            step_end = int(
                batch.num_computed_tokens_np[index] + batch.num_scheduled_tokens[index]
            )
            if plan.prompt:
                computed = int(batch.num_computed_tokens_np[index])
                lo = max(computed, plan.start)
                hi = min(
                    step_end,
                    plan.prompt_len - 1,
                    step_end if plan.end is None else plan.end,
                )
                if lo < hi:
                    offset = int(batch.query_start_loc_np[index]) + lo - computed
                    prompt_snapshots[request_id] = (
                        lo,
                        hidden_states[offset : offset + hi - lo].clone(),
                    )
            if not plan.generated:
                continue
            start, end = plan.select(step_end - int(last - first), step_end)
            if end <= start:
                continue
            offset = int(first) + start - (step_end - int(last - first))
            rows = sample_hidden_states[offset : offset + end - start].clone()
            snapshots[request_id] = (start, step_end, rows)
        self.pending = (
            PendingHiddenOutput(list(batch.req_ids), snapshots, prompt_snapshots)
            if snapshots or prompt_snapshots
            else None
        )

    def take_pending(self) -> PendingHiddenOutput | None:
        pending, self.pending = self.pending, None
        return pending
