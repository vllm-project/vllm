# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model runner for instances that only run the multi-modal encoder.

An encoder-only instance -- `--mm-encoder-only`, or the producer side of
encoder-cache disaggregation -- encodes the multi-modal items and publishes the
embeddings. It runs no language model: no KV cache, no sampler, no CUDA graphs.
"""

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.outputs import (
    ModelRunnerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.worker.gpu.buffer_utils import get_default_max_concurrency
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
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
        assert self.dp_size == 1, "An encoder-only instance does not support DP."

        depth = get_default_max_concurrency()
        # `UvaBufferPool` recycles a slot every `depth` steps and the device
        # reads the pooled host buffers in place. A sampling step is ordered by
        # the device wait in `AsyncOutput.get_output()`; this one never waits, so
        # it keeps one event per slot generation. Blocking: no driver busy-poll.
        self._input_reuse_events = [torch.Event(blocking=True) for _ in range(depth)]
        self._input_reuse_idx = 0

    def _wait_for_input_reuse(self) -> None:
        """Wait for the step that last wrote the slots this step will reuse."""
        self._input_reuse_events[self._input_reuse_idx].synchronize()

    def _mark_input_reuse(self) -> None:
        """Mark this step's last host write into the pooled input buffers."""
        self._input_reuse_events[self._input_reuse_idx].record()
        self._input_reuse_idx = (self._input_reuse_idx + 1) % len(
            self._input_reuse_events
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

        self._wait_for_input_reuse()
        self.update_pp_decode_requests()
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()
        if scheduler_output.total_num_scheduled_tokens == 0:
            self._mark_input_reuse()
            return self._no_forward(scheduler_output)

        batch_req_state, _ = self.gather_batch_req_state(scheduler_output, False)
        assert batch_req_state is not None
        # No CUDA graph, and no DP peer to agree a padded shape with.
        self.prepare_inputs(
            scheduler_output,
            batch_req_state,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=batch_req_state.num_tokens,
                num_reqs=None,
            ),
        )
        # Before the encoder, not after: it reads no pooled metadata, and
        # waiting on it costs ~22% throughput.
        self._mark_input_reuse()

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

        # `prepare_inputs_embeds` would build an inputs_embeds nobody reads and
        # raise "Encoder cache miss" for items this instance did not encode.
        with self.ec_connector.maybe_get_output(
            scheduler_output
        ) as ec_connector_output:
            self.model_state.execute_mm_encoder(scheduled_encoder_inputs)

        return ModelRunnerOutput.with_ec_conn_output(
            make_empty_encoder_model_runner_output(scheduler_output),
            ec_connector_output,
        )
