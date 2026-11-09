# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vllm.config.compilation import CompilationMode, CUDAGraphMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

ElasticPhase = Literal["full", "sliced", "auto"]


def elastic_phase(extra_args: dict | None) -> ElasticPhase:
    phase = (extra_args or {}).get("elastic_phase", "full")
    if phase not in ("full", "sliced", "auto"):
        raise ValueError("elastic_phase must be 'full', 'sliced', or 'auto'")
    return phase


@dataclass(frozen=True)
class SliceConfig:
    hidden_size: int
    moe_intermediate_size: int


@dataclass
class ElasticConfig:
    profiles: dict[str, SliceConfig]
    think_end_token_id: int

    @classmethod
    def from_vllm_config(cls, config: "VllmConfig") -> "ElasticConfig | None":
        hf = getattr(config.model_config, "hf_config", None)
        if hf is None or getattr(hf, "elastic_use_lazy_slicing", False) is not True:
            return None
        dimensions = getattr(hf, "elastic_thinking", {})
        full = SliceConfig(hf.hidden_size, hf.moe_intermediate_size)
        sliced = SliceConfig(
            dimensions.get("elastic_hidden_size", 0),
            dimensions.get("elastic_moe_intermediate_size", 0),
        )
        result = cls(
            {"full": full, "sliced": sliced},
            getattr(hf, "elastic_think_end_token_id", -1),
        )
        for child, parent in (
            (sliced.hidden_size, full.hidden_size),
            (sliced.moe_intermediate_size, full.moe_intermediate_size),
        ):
            if not isinstance(child, int) or not 0 < child <= parent or child % 64:
                raise ValueError(
                    "Elastic dimensions must be positive multiples of 64 "
                    "within the full dimensions"
                )
        if not 0 <= result.think_end_token_id < hf.vocab_size:
            raise ValueError("Elastic requires a valid elastic_think_end_token_id")
        parallel = config.parallel_config
        if full.moe_intermediate_size % parallel.tensor_parallel_size:
            raise ValueError(
                "Elastic full MoE intermediate size must be divisible by "
                "tensor parallel size"
            )
        if (
            getattr(hf, "use_bias", False)
            and hf.n_groups % parallel.tensor_parallel_size
        ):
            raise ValueError(
                "Elastic biased Mamba requires groups divisible by tensor parallel size"
            )

        if (
            hf.model_type != "nemotron_h"
            or str(config.model_config.dtype) != "torch.bfloat16"
            or config.quant_config is not None
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.enable_eplb
            or config.speculative_config is not None
            or config.lora_config is not None
            or config.kv_transfer_config is not None
            or config.scheduler_config.async_scheduling
            or config.compilation_config.mode != CompilationMode.NONE
            or config.compilation_config.cudagraph_mode
            not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
            or config.kernel_config.moe_backend != "triton"
            or getattr(hf, "moe_latent_size", None) is not None
        ):
            raise ValueError(
                "Elastic requires BF16 Nemotron-H, tensor parallelism only, "
                "Model Runner V2, "
                "synchronous scheduling, Triton MoE, compilation mode NONE, "
                "and CUDA graph mode NONE or FULL_DECODE_ONLY; "
                "quantization, LoRA, KV connectors and speculative decoding "
                "are not supported"
            )
        return result

    def request_phase(self, request: "Request") -> str:
        params = request.sampling_params
        phase = elastic_phase(params.extra_args if params else None)
        if phase == "auto":
            return (
                "full" if request.elastic_transition_position is not None else "sliced"
            )
        return phase

    def select_phase(self, requests: Iterable["Request"], previous: str | None) -> str:
        phase = "full"
        for request in requests:
            phase = self.request_phase(request)
            if phase != previous:
                break
        return phase
