# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the device work between the target forward and the draft
prefill of MTP/EAGLE speculative decoding, for model states that opt in
(``ModelState.low_latency_decode``).

After the target model's FULL CUDA graph, a decode step runs about a dozen small
kernels eagerly: the LM head, the rejection sampler, the request-state update and
the draft prefill's input preparation. At low batch their launch gaps leave the
GPU idle, so uniform decode steps whose sampling is fixed by the batch shape
capture that sequence once per shape and replay it.

``step_eligible`` decides per step, from host state every TP rank holds
identically; anything that changes the op sequence or reads host state at run
time (logprobs, grammar, logits processing, prefill, non-uniform batches) runs
eagerly. Sampling draws come from per-request seeds and positions, so a replay
draws what the eager kernels draw. The per-step tensors the region reads are
written into persistent buffers (``GPUModelRunner.prepare_inputs``), and the
graph key holds the address and layout of every tensor read, so a reallocation
captures a new graph instead of replaying a stale one. The first step of a key
runs eagerly, the second captures, later steps replay; the next replay waits for
the previous step's device-to-host output copy.
"""

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    get_max_chunk_logits,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states
from vllm.v1.worker.workspace import use_workspace_lane

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

logger = init_logger(__name__)

_PostForwardOutput = tuple[SamplerOutput, torch.Tensor, torch.Tensor]


def _tensor_key(t: torch.Tensor) -> tuple[Any, ...]:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def unsupported_reasons(runner: "GPUModelRunner") -> list[str]:
    """Configuration that rules the post-forward graphs out (empty if none)."""
    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

    speculator = runner.speculator
    sampler = runner.sampler
    rejection_sampler = runner.rejection_sampler
    unsupported = {
        "a non-CUDA platform": not current_platform.is_cuda(),
        "a speculator other than MTP/EAGLE": not isinstance(
            speculator, TargetDependentARSpeculator
        ),
        "a custom or missing rejection sampler": type(rejection_sampler)
        is not RejectionSampler,
        "watermarking": isinstance(sampler, GPUWatermarkSampler)
        or (
            isinstance(speculator, TargetDependentARSpeculator)
            and speculator.draft_watermarker is not None
        ),
        "VLLM_COMPUTE_NANS_IN_LOGITS": sampler is not None and sampler.compute_nans,
        "return_sampling_mask": sampler is not None and sampler.return_sampling_mask,
        "pipeline parallelism": runner.use_pp,
        "prefill/decode context parallelism": runner.pcp_manager is not None
        or runner.dcp_size > 1,
        "batch-sharded sampling": runner.batch_sharder is not None,
        "adaptive verification": runner.adaptive_verification is not None,
        "LoRA": bool(runner.lora_config),
        # A KV connector is allowed: its per-step work (start_load_kv before the
        # forward, post_forward after the drafter) lies outside the region.
        "the aux output connector": runner.aux_output_connector is not None,
        "KV-sharing fast prefill": runner.fast_prefill is not None,
        "trace replay": sampler is not None and sampler.trace_replay_state is not None,
    }
    return [name for name, hit in unsupported.items() if hit]


def maybe_create_post_forward_cudagraphs(
    runner: "GPUModelRunner",
) -> "PostForwardCudaGraphs | None":
    """The runner's post-forward graphs, or None when the model state does not
    opt in or the configuration is unsupported (logged)."""
    if not runner.model_state.low_latency_decode or runner.speculator is None:
        return None
    reasons = unsupported_reasons(runner)
    if reasons:
        logger.warning(
            "The post-forward CUDA graph is not supported with %s; the "
            "post-forward region runs eagerly.",
            ", ".join(reasons),
        )
        return None
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
        TargetDependentARSpeculator,
    )

    speculator = runner.speculator
    assert isinstance(speculator, TargetDependentARSpeculator)
    logger.info("Capturing the post-forward region of MTP decode steps.")
    return PostForwardCudaGraphs(
        speculator, runner.max_num_reqs, runner.max_num_tokens, runner.device
    )


class PostForwardCudaGraphs:
    """Per-shape CUDA graphs of the post-forward region, and the persistent
    per-step inputs they read."""

    # Uniform decode yields one key per batch size; more keys than this means
    # addresses move, and the further shapes run eagerly.
    MAX_GRAPHS = 32

    def __init__(
        self,
        speculator: "TargetDependentARSpeculator",
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.speculator = speculator
        int32, int64 = torch.int32, torch.int64
        self.idx_mapping = torch.zeros(max_num_reqs, dtype=int64, device=device)
        self.cu_num_logits = torch.zeros(max_num_reqs + 1, dtype=int32, device=device)
        self.expanded_idx_mapping = torch.zeros(
            max_num_tokens, dtype=int64, device=device
        )
        self.expanded_local_pos = torch.zeros(
            max_num_tokens, dtype=int32, device=device
        )
        self.logits_indices = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )

        self.pool: Any = None  # A platform graph pool handle.
        self.graphs: dict[Hashable, tuple[torch.cuda.CUDAGraph, Any]] = {}
        self.warmed_up: set[Hashable] = set()
        # Recorded after the async output copy of the last replayed step.
        self.output_copy_event: torch.cuda.Event | None = None
        self.num_replays = 0
        self.warned_limit = False

    def step_eligible(
        self,
        runner: "GPUModelRunner",
        input_batch: "InputBatch",
        grammar_output: "GrammarOutput | None",
        full_cudagraph: bool,
    ) -> bool:
        """Whether this step's post-forward region has the captured op sequence.

        Reads only host state that every TP rank holds identically."""
        num_reqs = input_batch.num_reqs
        if not full_cudagraph or num_reqs == 0 or input_batch.has_prefill:
            return False
        if grammar_output is not None or input_batch.has_structured_output_reqs:
            return False
        num_steps = runner.num_speculative_steps
        if input_batch.num_draft_tokens != num_reqs * num_steps or not np.all(
            input_batch.num_scheduled_tokens == num_steps + 1
        ):
            return False
        if input_batch.num_tokens > get_max_chunk_logits(runner.vocab_size):
            return False
        sampler = runner.sampler
        prompt_logprobs = runner.prompt_logprobs_worker
        assert sampler is not None and prompt_logprobs is not None
        idx_mapping_np = input_batch.idx_mapping_np
        return not (
            np.any(sampler.needs_logits_processing[idx_mapping_np])
            or sampler.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
            or np.any(prompt_logprobs.uses_prompt_logprobs[idx_mapping_np])
            or prompt_logprobs.token_id_scores
            or self.speculator.acceptance_estimator is not None
        )

    def run(
        self,
        runner: "GPUModelRunner",
        input_batch: "InputBatch",
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> _PostForwardOutput:
        """Sample, update the request states and prepare the draft inputs, as
        the eager path does, by running or replaying this shape's graph."""
        drafter_hidden_states = get_drafter_hidden_states(runner.model, hidden_states)
        assert runner.sampler is not None
        sampling_states = runner.sampler.sampling_states
        # These host-backed buffers rotate through a small pool as the host
        # stages writes (every step for the sampling states), so they key the
        # graph too: each pool slot gets its own graph.
        temperature = sampling_states.temperature.gpu
        seeds = sampling_states.seeds.gpu
        prefill_len = runner.req_states.prefill_len.gpu
        key = (
            input_batch.num_reqs,
            input_batch.num_tokens,
            *(
                _tensor_key(t)
                for t in (
                    hidden_states,
                    drafter_hidden_states,
                    *(aux_hidden_states or ()),
                    input_batch.idx_mapping,
                    input_batch.cu_num_logits,
                    input_batch.expanded_idx_mapping,
                    input_batch.expanded_local_pos,
                    input_batch.logits_indices,
                    input_batch.input_ids,
                    input_batch.positions,
                    input_batch.query_start_loc,
                    input_batch.seq_lens,
                    temperature,
                    seeds,
                    prefill_len,
                )
            ),
        )

        def post_forward() -> _PostForwardOutput:
            sampler_output, num_sampled, num_rejected = runner.sample(
                hidden_states, input_batch, None
            )
            runner.postprocess_sampled(
                input_batch.idx_mapping,
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch.query_start_loc,
            )
            with use_workspace_lane(runner._draft_workspace_lane):
                self.speculator.prepare_draft_inputs(
                    input_batch,
                    drafter_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    runner.req_states.last_sampled_tokens,
                    runner.req_states.next_prefill_tokens,
                    temperature,
                    seeds,
                )
            return sampler_output, num_sampled, num_rejected

        outputs = self._run(key, post_forward)
        self.speculator.inputs_prepared = True
        return outputs

    def _run(
        self, key: Hashable, fn: Callable[[], _PostForwardOutput]
    ) -> _PostForwardOutput:
        """Run ``fn`` eagerly (first step of a key, or past the graph limit),
        capture it (second step), or replay its graph."""
        entry = self.graphs.get(key)
        if entry is None:
            if key not in self.warmed_up:
                # Only warmed-up keys capture, so this bounds the graphs.
                if len(self.warmed_up) < self.MAX_GRAPHS:
                    self.warmed_up.add(key)
                elif not self.warned_limit:
                    self.warned_limit = True
                    logger.warning(
                        "%d post-forward batch shapes seen; later shapes run eagerly.",
                        self.MAX_GRAPHS,
                    )
                return fn()
            if self.pool is None:
                self.pool = current_platform.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            # Captured while serving: other threads may use CUDA meanwhile.
            with torch.cuda.graph(
                graph,
                pool=self.pool,
                stream=current_stream(),
                capture_error_mode="thread_local",
            ):
                outputs = fn()
            entry = self.graphs[key] = (graph, outputs)
        graph, outputs = entry
        if self.output_copy_event is not None:
            # The previous replay's outputs may still be in the copy to host.
            current_stream().wait_event(self.output_copy_event)
        graph.replay()
        self.num_replays += 1
        return outputs

    def set_output_copy_event(self, event: torch.cuda.Event) -> None:
        """Record the event after which this step's outputs are on the host."""
        self.output_copy_event = event

    def release(self) -> None:
        """Drop the graphs, e.g. before the buffers they read are reallocated."""
        self.graphs.clear()
        self.warmed_up.clear()
        self.output_copy_event = None
        self.warned_limit = False
