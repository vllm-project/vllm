# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model runner for instances that only run the multi-modal encoder.

An encoder-only instance -- `--mm-encoder-only`, or the producer side of
encoder-cache disaggregation -- encodes the multi-modal items and publishes the
embeddings for a peer to consume. It runs no language model, holds no KV cache
and samples no token, so most of a step does not apply to it: no attention
metadata, no forward pass, no sampler, no CUDA graphs.

Keeping that here rather than as `is_encoder_only` branches inside the shared
runner keeps the exceptions in one place, and keeps invariants that only hold
for a full step (see `execute_model` below) from being read as universal.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.outputs import (
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.lora_utils import get_num_active_loras_for_dispatch
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class MMEncoderModelRunner(GPUModelRunner):
    """Encoder-only variant of the V2 GPU model runner."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        assert self.supports_mm_inputs, (
            "An encoder-only instance must serve a multi-modal model."
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def capture_model(self) -> int:
        return 0

    def _dummy_run(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        empty = torch.empty(0, device=self.device)
        return empty, empty

    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        return

    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        return

    def _no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return self._merge_ec_connector_no_forward(
            scheduler_output, self.kv_connector.no_forward(scheduler_output)
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ) -> ModelRunnerOutput:
        assert not dummy_run, "An encoder-only instance runs no dummy batch."

        self.update_pp_decode_requests()
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return self._no_forward(scheduler_output)

        batch_req_state, uniform_tok_count = self.gather_batch_req_state(
            scheduler_output, False
        )
        assert batch_req_state is not None
        num_active_loras = 0
        if self.lora_config:
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config,
                self.lora_state,
                list(scheduler_output.num_scheduled_tokens.keys()),
                False,
            )
        # This rank runs no compiled graph, but the DP peers size their padding
        # from the shape agreed here, so it still has to take part.
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            len(scheduler_output.num_scheduled_tokens),
            batch_req_state.num_tokens,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            max_query_len=max(scheduler_output.num_scheduled_tokens.values()),
            num_active_loras=num_active_loras,
        )
        if batch_desc.num_tokens == 0:
            return self._no_forward(scheduler_output)

        self.prepare_inputs(scheduler_output, batch_req_state, batch_desc)

        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if self.lora_config is not None:
            set_active_mm_loras(
                model=self.model,
                lora_manager=self.lora_manager,
                encoder_cache=self.encoder_cache,
                req_id_to_index=self.req_states.req_id_to_index,
                lora_state=self.lora_state,
                scheduled_encoder_inputs=scheduled_encoder_inputs,
            )

        # Encode and publish, nothing else. `prepare_inputs_embeds` would build
        # an inputs_embeds nobody reads, and would raise "Encoder cache miss"
        # for any scheduled item this instance did not encode.
        with self.ec_connector.maybe_get_output(
            scheduler_output
        ) as ec_connector_output:
            self.model_state.execute_mm_encoder(scheduled_encoder_inputs)

        # NOTE: This output is built on the host and carries no sampled token,
        # so unlike a full step it never waits on the device. Anything the
        # device still reads from a recycled host buffer must be ordered here
        # explicitly -- a sampling step gets that ordering from its output copy.
        return ModelRunnerOutput.with_ec_conn_output(
            make_empty_encoder_model_runner_output(scheduler_output),
            ec_connector_output,
        )
