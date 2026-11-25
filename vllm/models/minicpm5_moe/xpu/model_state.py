# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM5's model-owned runner state."""

from functools import partial

import torch

from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from .specialized_decode.config import configure


def _active_xpu_capture_stream(
    microbatch_stream: torch.xpu.Stream | None, desc: BatchExecutionDescriptor
) -> torch.xpu.Stream:
    if desc.num_ubatches > 1:
        assert microbatch_stream is not None
        return microbatch_stream
    return torch.xpu.current_stream()


def _bind_capture_stream(manager: CudaGraphManager) -> None:
    # Both profiling and serving create fresh graph managers. The XPU stream
    # context does not maintain the CUDA thread-local stream cache.
    ubatches = manager.ubatch_runner
    manager._capture_stream = partial(
        _active_xpu_capture_stream,
        ubatches.capture_stream if ubatches is not None else None,
    )


class MiniCPM5XpuModelState(DefaultModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.decode = None
        self.short_fa2 = None
        self.candidates = None
        self.dspark_draft = None
        extra = vllm_config.additional_config
        config = configure(vllm_config)
        if config is not None:
            from .prefill import configure_prefill_attention
            from .specialized_decode.decode import SpecializedDecode

            configure_prefill_attention(model.model, vllm_config)
            spec = vllm_config.speculative_config
            options = extra if isinstance(extra, dict) else {}
            mono_draft = options.get("minicpm5_dspark_mono_draft", False)
            self.dspark_shard_sampling = options.get(
                "minicpm5_dspark_shard_sampling", False
            )
            self.dspark_pruned_markov = options.get(
                "minicpm5_dspark_pruned_markov", False
            )
            if type(self.dspark_shard_sampling) is not bool:
                raise ValueError("minicpm5_dspark_shard_sampling must be a boolean")
            if type(self.dspark_pruned_markov) is not bool:
                raise ValueError("minicpm5_dspark_pruned_markov must be a boolean")
            if type(mono_draft) is not bool:
                raise ValueError("minicpm5_dspark_mono_draft must be a boolean")
            draft_slots = 0
            if (
                mono_draft
                and spec is not None
                and spec.method == "dspark"
                and spec.num_speculative_tokens == 7
                and config.tp in (1, 2)
                and vllm_config.parallel_config.decode_context_parallel_size == 1
                and vllm_config.parallel_config.prefill_context_parallel_size == 1
                and spec.draft_parallel_config.tensor_parallel_size == config.tp
            ):
                draft_slots = spec.draft_model_config.hf_config.num_hidden_layers
            self.decode = SpecializedDecode(model.model, config, draft_slots)
            self.decode.attach(model.model)
            if config.pre_o_fa2_adaptive_split1:
                from .specialized_decode.short_fa2_graph import ShortFa2Selection

                self.short_fa2 = ShortFa2Selection()
            from .specialized_decode.sampling import Candidates

            self.candidates = Candidates(vllm_config, self.decode, self.max_num_reqs)
            head = model.lm_head
            if head.num_embeddings_padded != head.org_vocab_size:
                self.candidates.enabled = False

    def prepare_inputs(self, input_batch, req_states):
        if self.candidates is not None:
            self.candidates.batch = input_batch
        return super().prepare_inputs(input_batch, req_states)

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        if self.candidates is not None:
            self.candidates.batch = None
        return super().prepare_dummy_inputs(num_reqs, num_tokens)

    def custom_sampler(self, sampler):
        if self.candidates is None or not self.candidates.enabled:
            return None
        from .specialized_decode.sampling import CandidateSampler

        return CandidateSampler(sampler, self.candidates), None

    def configure_dspark_draft(self, draft_model):
        from .draft_attention import configure_draft_attention

        configure_draft_attention(draft_model, self.vllm_config)
        if (
            self.decode is None
            or not self.decode.draft_slots
            or self.dspark_draft is not None
        ):
            return
        from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

        from .specialized_decode.draft import DraftMono

        if isinstance(draft_model, Qwen3DSparkForCausalLM):
            self.dspark_draft = DraftMono(
                draft_model.model,
                self.decode,
                self.decode.config.max_context_len,
                self.max_num_reqs,
                tp=self.decode.config.tp,
            )
            if self.dspark_shard_sampling:
                self.dspark_draft.configure_sampling(draft_model)
                if self.dspark_draft.markov_shard is not None:
                    draft_model.sample_sequential = self.dspark_draft.sample_sequential
            # Gather the base logits once when sampling the full vocabulary.
            if self.dspark_pruned_markov and self.dspark_draft.markov_shard is None:
                from .specialized_decode.markov import PrunedMarkovSampler

                sampler = PrunedMarkovSampler.from_model(draft_model)
                if sampler is not None:
                    draft_model.sample_sequential = sampler
            self.dspark_draft.attach(draft_model.model)

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        if self.decode is not None:
            self.decode.bind_caches()
        if self.dspark_draft is not None:
            self.dspark_draft.bind_caches()

    def configure_xpu_graphs(self, manager, speculator) -> None:
        spec = self.vllm_config.speculative_config
        dspark = spec is not None and spec.method == "dspark" and speculator is not None
        options = self.vllm_config.additional_config
        if self.decode is None and not (
            dspark
            and isinstance(options, dict)
            and options.get("minicpm5_dspark_split_kv") is True
        ):
            return
        _bind_capture_stream(manager)
        if dspark:
            draft_manager = speculator.query_cudagraph_manager
            if draft_manager is not None:
                _bind_capture_stream(draft_manager)
        if self.short_fa2 is not None:
            from .specialized_decode.short_fa2_graph import attach_short_fa2_graph

            attach_short_fa2_graph(manager, self.decode, self.short_fa2)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
        if self.short_fa2 is not None:
            self.short_fa2.prepare(input_batch, cudagraph_mode, for_capture)
        if self.decode is not None:
            draft_tokens = self.decode.config.speculative_tokens
            eligible = not input_batch.has_prefill and bool(
                (
                    (input_batch.num_scheduled_tokens >= 1)
                    & (input_batch.num_scheduled_tokens <= 1 + draft_tokens)
                ).all()
            )
            for item in metadata.values():
                item.minicpm5_decode_eligible = eligible
                item.minicpm5_verify_eligible = eligible and draft_tokens > 0
        return metadata

    def compute_sample_logits(self, hidden_states):
        candidates = self.candidates
        k = (
            None
            if candidates is None
            else candidates.plan(hidden_states.shape[0], None)
        )
        if candidates is None or k is None:
            logits = self.model.compute_logits(hidden_states)
        else:
            shard = self.model.compute_logits_local(hidden_states)
            logits = candidates.logits(shard, k)
        if self.decode is not None:
            self.decode.after_step()
        return logits
