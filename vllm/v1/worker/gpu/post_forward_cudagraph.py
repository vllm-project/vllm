# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the device work between the target forward and the DSpark
draft step on decode steps, for model states that offer them.

A model state opts in with a true ``post_forward_cudagraphs`` attribute; the
others, and runners without a DSpark speculator, keep the eager region.

After the target model's FULL CUDA graph, a decode step runs about two dozen
small kernels eagerly before the draft step's graphs: the hidden-state gather
and LM head (with its all-gather), the rejection sampler, the request-state
update (``postprocess_sampled``) and the draft input preparation. At small
batch sizes their launches leave the GPU idle between them. For uniform decode
steps whose sampling is fixed by the batch shape, this region is captured in
one CUDA graph per shape and replayed.

Eligibility is decided per step on the host, from state every tensor-parallel
rank holds identically. Anything that changes the op sequence or reads host
state at run time keeps the eager path: logprobs, prompt logprobs, grammar,
logits processing, prefill and non-uniform batches. Sampling draws come from
per-request seeds and positions, not from a host-advanced generator, so a
replay draws what the eager kernels draw.

A graph reads its inputs at the addresses it was captured with. The per-step
tensors ``prepare_inputs`` allocates (``idx_mapping``, ``cu_num_logits``, the
expanded mapping and positions, ``logits_indices``) are copied into buffers
owned here, in one launch. So are the host-backed temperature, seed and
prefill-length buffers, which rotate through a small pool of host buffers; the
region reads the copies. The other inputs are persistent runner, sampler and
speculator buffers or the target graph's outputs. The key holds the address of
each of them, so a moved buffer captures a new graph instead of replaying a
stale one. The first step of a key runs eagerly (it compiles kernels and warms
up allocations), the second captures, later ones replay.

The graph outputs (sampled tokens, counts) are static. The previous step's
async output copy may still read them on the output copy stream, so a replay
waits for that stream first.

The draft context work that reads only the target hidden states (the combined
hidden states and, if the draft model supports it, every draft layer's context
KV projection) is forked onto the speculator's side stream after the LM head
and joined at the end of the draft input preparation, so it runs beside
sampling. Eager or captured, the fork and join are stream waits inside the
region.

The graphs are captured while serving, into a pool of their own that the
startup memory profile does not see. They share that pool, so it holds about
one region's temporaries at the largest batch (mostly its logits).
"""

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
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
    from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)

PostForwardOutput = tuple[SamplerOutput, torch.Tensor, torch.Tensor]


@triton.jit
def _copy(src, dst, n, BLOCK: tl.constexpr):
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        tl.store(dst + offs, tl.load(src + offs, mask=offs < n), mask=offs < n)


@triton.jit
def _copy_step_tensors_kernel(
    idx_mapping,
    idx_mapping_out,
    cu_num_logits,
    cu_num_logits_out,
    expanded_idx_mapping,
    expanded_idx_mapping_out,
    expanded_local_pos,
    expanded_local_pos_out,
    logits_indices,
    logits_indices_out,
    temperature,
    temperature_out,
    seeds,
    seeds_out,
    prefill_len,
    prefill_len_out,
    num_reqs,
    num_logits,
    num_states,
    BLOCK: tl.constexpr,
):
    # One program per tensor.
    tensor = tl.program_id(0)
    if tensor == 0:
        _copy(idx_mapping, idx_mapping_out, num_reqs, BLOCK)
    elif tensor == 1:
        _copy(cu_num_logits, cu_num_logits_out, num_reqs + 1, BLOCK)
    elif tensor == 2:
        _copy(expanded_idx_mapping, expanded_idx_mapping_out, num_logits, BLOCK)
    elif tensor == 3:
        _copy(expanded_local_pos, expanded_local_pos_out, num_logits, BLOCK)
    elif tensor == 4:
        _copy(logits_indices, logits_indices_out, num_logits, BLOCK)
    elif tensor == 5:
        _copy(temperature, temperature_out, num_states, BLOCK)
    elif tensor == 6:
        _copy(seeds, seeds_out, num_states, BLOCK)
    else:
        _copy(prefill_len, prefill_len_out, num_states, BLOCK)


def _tensor_key(t: torch.Tensor) -> tuple[Any, ...]:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


@contextmanager
def _reading_copies(
    tensors: tuple["UvaBackedTensor", ...], copies: tuple[torch.Tensor, ...]
) -> Iterator[None]:
    """Point each host-backed tensor's ``gpu`` view at its device copy."""
    views = [t.gpu for t in tensors]
    for t, copy in zip(tensors, copies, strict=True):
        t.gpu = copy
    try:
        yield
    finally:
        for t, view in zip(tensors, views, strict=True):
            t.gpu = view


class PostForwardCudaGraphs:
    """Per-shape CUDA graphs of the post-forward region and the buffers of the
    per-step tensors they read."""

    # Uniform decode gives one key per batch size; past this many keys,
    # further shapes run eagerly.
    MAX_GRAPHS = 32

    @classmethod
    def create(cls, runner: "GPUModelRunner") -> "PostForwardCudaGraphs | None":
        """The runner's post-forward graphs for DSpark, when its model state
        offers them (``post_forward_cudagraphs``), or None."""
        from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        speculator = runner.speculator
        if not getattr(
            runner.model_state, "post_forward_cudagraphs", False
        ) or not isinstance(speculator, DSparkSpeculator):
            return None
        sampler = runner.sampler
        unsupported = {
            "a non-CUDA platform": not current_platform.is_cuda(),
            "a custom rejection sampler": type(runner.rejection_sampler)
            is not RejectionSampler,
            "watermarking": runner.vllm_config.watermark_config is not None,
            "VLLM_COMPUTE_NANS_IN_LOGITS": sampler is None or sampler.compute_nans,
            "return_sampling_mask": sampler is None or sampler.return_sampling_mask,
            "pipeline or context parallelism": runner.use_pp
            or runner.pcp_manager is not None
            or runner.dcp_size > 1,
            "batch-sharded sampling": runner.batch_sharder is not None,
            "adaptive verification": runner.adaptive_verification is not None,
            "LoRA": runner.lora_config is not None,
            "a KV connector": runner.kv_connector is not NO_OP_KV_CONNECTOR,
            "an aux output connector": runner.aux_output_connector is not None,
            "KV-sharing fast prefill": runner.fast_prefill is not None,
        }
        reasons = [name for name, hit in unsupported.items() if hit]
        if reasons:
            logger.warning(
                "The DSpark post-forward region runs eagerly with %s.",
                ", ".join(reasons),
            )
            return None
        return cls(
            speculator, runner.max_num_reqs, runner.max_num_tokens, runner.device
        )

    def __init__(
        self,
        speculator: "DSparkSpeculator",
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
    ):
        self.speculator = speculator
        speculator.init_context_side_stream()
        # The dtypes prepare_inputs gives these tensors.
        int32, int64 = torch.int32, torch.int64
        self.idx_mapping = torch.zeros(max_num_reqs, dtype=int64, device=device)
        self.cu_num_logits = torch.zeros(max_num_reqs + 1, dtype=int32, device=device)
        self.expanded_idx_mapping = torch.zeros(
            max_num_tokens, dtype=int64, device=device
        )
        self.expanded_local_pos = torch.zeros(
            max_num_tokens, dtype=int32, device=device
        )
        self.logits_indices = torch.zeros(max_num_tokens, dtype=int64, device=device)
        # Per request state, as the sampler and the request states hold them.
        self.temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        self.seeds = torch.zeros(max_num_reqs, dtype=int64, device=device)
        self.prefill_len = torch.zeros(max_num_reqs, dtype=int32, device=device)
        # A platform graph pool handle, created at the first capture.
        self.pool: Any = None
        self.graphs: dict[Hashable, tuple[torch.cuda.CUDAGraph, PostForwardOutput]]
        self.graphs = {}
        self.warmed_up: set[Hashable] = set()

    def _step_eligible(
        self,
        runner: "GPUModelRunner",
        input_batch: "InputBatch",
        hidden_states: torch.Tensor,
        grammar_output: "GrammarOutput | None",
    ) -> bool:
        """Whether this step has the captured op sequence."""
        num_reqs = input_batch.num_reqs
        num_steps = runner.num_speculative_steps
        # Only a FULL target graph returns its static output buffer; eager
        # steps would churn keys.
        manager = runner.cudagraph_manager
        static_hidden_states = None if manager is None else manager.hidden_states
        if (
            static_hidden_states is None
            or hidden_states.data_ptr() != static_hidden_states.data_ptr()
            or num_reqs == 0
            or input_batch.has_prefill
            or grammar_output is not None
            or input_batch.has_structured_output_reqs
            or input_batch.num_draft_tokens != num_reqs * num_steps
            or not np.all(input_batch.num_scheduled_tokens == num_steps + 1)
            or input_batch.num_tokens > get_max_chunk_logits(runner.vocab_size)
        ):
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
        )

    def run(
        self,
        runner: "GPUModelRunner",
        input_batch: "InputBatch",
        hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        grammar_output: "GrammarOutput | None",
    ) -> PostForwardOutput | None:
        """Sample, update the request states and prepare the draft inputs as
        the eager path does, by running or replaying this shape's graph. None
        if the step is not eligible."""
        if not self._step_eligible(runner, input_batch, hidden_states, grammar_output):
            return None
        speculator = self.speculator
        sampler = runner.sampler
        rejection_sampler = runner.rejection_sampler
        assert sampler is not None and rejection_sampler is not None

        num_reqs = input_batch.num_reqs
        num_logits = input_batch.logits_indices.shape[0]
        # The host-backed per-request state the region reads.
        host_backed = (
            sampler.sampling_states.temperature,
            sampler.sampling_states.seeds,
            runner.req_states.prefill_len,
        )
        num_states = host_backed[0].gpu.shape[0]
        assert all(t.gpu.shape[0] == num_states for t in host_backed)
        temperature = self.temperature[:num_states]
        seeds = self.seeds[:num_states]
        prefill_len = self.prefill_len[:num_states]
        _copy_step_tensors_kernel[(8,)](
            input_batch.idx_mapping,
            self.idx_mapping,
            input_batch.cu_num_logits,
            self.cu_num_logits,
            input_batch.expanded_idx_mapping,
            self.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.expanded_local_pos,
            input_batch.logits_indices,
            self.logits_indices,
            host_backed[0].gpu,
            temperature,
            host_backed[1].gpu,
            seeds,
            host_backed[2].gpu,
            prefill_len,
            num_reqs,
            num_logits,
            num_states,
            BLOCK=1024,
        )
        batch = replace(
            input_batch,
            idx_mapping=self.idx_mapping[:num_reqs],
            cu_num_logits=self.cu_num_logits[: num_reqs + 1],
            expanded_idx_mapping=self.expanded_idx_mapping[:num_logits],
            expanded_local_pos=self.expanded_local_pos[:num_logits],
            logits_indices=self.logits_indices[:num_logits],
        )
        drafter_hidden_states = get_drafter_hidden_states(runner.model, hidden_states)
        key = (
            num_reqs,
            input_batch.num_tokens,
            *(
                _tensor_key(t)
                for t in (
                    hidden_states,
                    drafter_hidden_states,
                    *(aux_hidden_states or ()),
                    batch.input_ids,
                    batch.positions,
                    batch.query_start_loc,
                    batch.seq_lens,
                    temperature,
                    seeds,
                    prefill_len,
                )
            ),
        )

        def post_forward() -> PostForwardOutput:
            logits = runner.model.compute_logits(hidden_states[batch.logits_indices])
            # The draft context work that reads only the target hidden states
            # overlaps sampling; prepare_draft_inputs joins it.
            with use_workspace_lane(runner._draft_workspace_lane):
                speculator.fork_target_context(
                    batch, drafter_hidden_states, aux_hidden_states
                )
            sampler_output = rejection_sampler(logits, batch, speculator.draft_logits)
            num_sampled = sampler_output.num_sampled
            num_rejected = sampler_output.num_rejected
            runner.postprocess_sampled(
                batch.idx_mapping,
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                batch.query_start_loc,
            )
            with use_workspace_lane(runner._draft_workspace_lane):
                speculator.prepare_draft_inputs(
                    batch,
                    drafter_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    runner.req_states.last_sampled_tokens,
                    runner.req_states.next_prefill_tokens,
                    temperature,
                    seeds,
                    context_forked=True,
                )
            return sampler_output, num_sampled, num_rejected

        # Eager or captured, the region reads the copies.
        with _reading_copies(host_backed, (temperature, seeds, prefill_len)):
            outputs = self._run(key, post_forward, runner.output_copy_stream)
        speculator.inputs_prepared = True
        return outputs

    def _run(
        self,
        key: Hashable,
        fn: Callable[[], PostForwardOutput],
        output_copy_stream: torch.cuda.Stream,
    ) -> PostForwardOutput:
        """Run ``fn`` eagerly (first step of a key, or past the graph limit),
        capture it (second step), or replay its graph."""
        entry = self.graphs.get(key)
        if entry is None:
            if key not in self.warmed_up:
                if len(self.warmed_up) < self.MAX_GRAPHS:
                    self.warmed_up.add(key)
                else:
                    logger.warning_once(
                        "Over %d post-forward batch shapes; new ones run eagerly.",
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
        # The previous step's outputs may still be in the copy to the host.
        current_stream().wait_stream(output_copy_stream)
        graph.replay()
        return outputs
