# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import replace
from typing import Any

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)

logger = init_logger(__name__)

_DCUT_FALLBACK_RATIO = 3 / 4
_DCUT_RATIO_NUMS = (1, 2, 3, 4)
_DCUT_PROFILE_SEQ_LEN = 2048
_DCUT_PROFILE_WARMUPS = 3
_DCUT_PROFILE_STEPS = 5


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        self._runner = runner

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None

        self.dflash_causal = self.dflash_config.get("causal", False)
        self._dcut_keep_lens_cache: torch.Tensor | None = None
        self._dcut_costs_by_bs: dict[int, torch.Tensor] = {}

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        # The draft model is text-only — clear the target's multimodal
        # flag so flash_attn is not rejected for mm_prefix support.
        arch = base.model_config.model_arch_config
        if arch.is_mm_prefix_lm:
            base.model_config.model_arch_config = replace(arch, is_mm_prefix_lm=False)
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=not self.dflash_causal,
            ),
        )

    @override
    def _warn_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        pass

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._dcut_keep_lens_cache = None
        if self.speculative_config.dflash_dcut_mode == "off":
            return super()._sample_draft_tokens(hidden_states, sampling_metadata)

        logits = self.model.compute_logits(hidden_states)
        draft_token_ids, draft_probs = self._sample_from_logits(
            logits, sampling_metadata
        )
        token_logprobs = (
            logits.gather(1, draft_token_ids.unsqueeze(1)).squeeze(1).float()
            - torch.logsumexp(logits, dim=-1).float()
            if draft_probs is None
            else torch.log(
                draft_probs.gather(1, draft_token_ids.unsqueeze(1)).squeeze(1)
            )
        )

        self._dcut_keep_lens_cache = self._select_dcut_keep_lens(
            token_logprobs.view(-1, self.num_speculative_tokens)
        )
        return draft_token_ids, draft_probs

    def _select_dcut_keep_lens(self, logprobs: torch.Tensor) -> torch.Tensor:
        bs, num_draft_tokens = logprobs.shape
        cumlogprobs = logprobs.cumsum(dim=1).flatten()
        profile_bs = min((x for x in self._dcut_costs_by_bs if x >= bs), default=None)
        if profile_bs is None:
            dcut = self.speculative_config.dflash_dcut
            if dcut == "auto":
                logger.warning_once(
                    "DFlash D-Cut selector has no profiled cost for bs=%d; "
                    "fallback ratio %.2f.",
                    bs,
                    _DCUT_FALLBACK_RATIO,
                )
                dcut = _DCUT_FALLBACK_RATIO
            num_keep_draft_tokens = self._get_dcut_keep_count(
                bs, num_draft_tokens, float(dcut)
            )
            _, top_indices = torch.topk(cumlogprobs, k=num_keep_draft_tokens)
            updates = torch.ones_like(top_indices, dtype=torch.int32)
        else:
            keep_counts = self._dcut_keep_counts[bs]
            costs = self._dcut_costs_by_bs[profile_bs]
            sorted_logprobs, top_indices = torch.sort(cumlogprobs, descending=True)
            prefix_scores = torch.cumsum(torch.exp(sorted_logprobs), dim=0)
            candidate_scores = torch.zeros_like(costs)
            valid = keep_counts > 0
            candidate_scores[valid] = prefix_scores[keep_counts[valid] - 1]
            num_keep_draft_tokens = keep_counts[torch.argmax(candidate_scores / costs)]
            updates = (
                torch.arange(bs * num_draft_tokens, device=self.device)
                < num_keep_draft_tokens
            ).to(torch.int32)
        keep_lens = torch.zeros((bs,), dtype=torch.int32, device=self.device)
        keep_lens.scatter_add_(0, top_indices // num_draft_tokens, updates)
        return keep_lens

    @staticmethod
    def _get_dcut_keep_count(bs: int, num_draft_tokens: int, ratio: float) -> int:
        return max(0, math.ceil(bs * (num_draft_tokens + 1) * ratio) - bs)

    def take_dcut_keep_lens(self) -> torch.Tensor | None:
        return self._dcut_keep_lens_cache

    def profile_dcut_cost_table(self) -> None:
        costs_by_bs: dict[int, list[tuple[int, float]]] = {}
        max_bs = min(self.max_batch_size, self._runner.max_num_reqs)
        bs_range = torch.arange(max_bs + 1, device=self.device, dtype=torch.long)[
            :, None
        ]
        ratio_nums = torch.tensor(
            _DCUT_RATIO_NUMS, device=self.device, dtype=torch.long
        )
        keep_counts = torch.div(
            bs_range * (self.num_speculative_tokens + 1) * ratio_nums + 3,
            4,
            rounding_mode="floor",
        )
        self._dcut_keep_counts = torch.clamp(keep_counts - bs_range, min=0)
        for bs in self._get_dcut_profile_batch_sizes():
            entries: list[tuple[int, float]] = []
            full_draft_tokens = bs * (1 + self.num_speculative_tokens)
            self._dcut_costs_by_bs[bs] = torch.ones(
                len(_DCUT_RATIO_NUMS), dtype=torch.float32, device=self.device
            )
            for keep_count in self._dcut_keep_counts[bs].tolist():
                target_tokens = bs + keep_count
                cost = self._profile_dcut_full_cost_ms(
                    bs=bs,
                    target_tokens=target_tokens,
                    draft_tokens=full_draft_tokens,
                )
                entries.append((keep_count, cost))
            costs_by_bs[bs] = entries
            self._dcut_costs_by_bs[bs] = torch.tensor(
                [x[1] for x in entries], dtype=torch.float32, device=self.device
            )
        if costs_by_bs:
            logger.info("DFlash D-Cut warmup full-cost table: %s", costs_by_bs)

    def _get_dcut_profile_batch_sizes(self) -> tuple[int, ...]:
        full_tokens_per_req = 1 + self.num_speculative_tokens
        max_bs = min(self.max_batch_size, self._runner.max_num_reqs)
        capture_sizes = self.compilation_config.cudagraph_capture_sizes or []
        sizes = [
            capture_size // full_tokens_per_req
            for capture_size in capture_sizes
            if capture_size % full_tokens_per_req == 0
            and 0 < capture_size // full_tokens_per_req <= max_bs
        ]
        sizes.append(max_bs)
        return tuple(sorted(set(sizes)))

    def _profile_dcut_full_cost_ms(
        self, *, bs: int, target_tokens: int, draft_tokens: int
    ) -> float:
        profile_seq_lens = min(
            _DCUT_PROFILE_SEQ_LEN,
            self._runner.max_model_len,
        )
        dummy_run_kwargs = dict(
            force_attention=True,
            allow_microbatching=False,
            skip_eplb=True,
            is_profile=False,
            dcut_profile_num_reqs=bs,
            drafter_dummy_num_tokens=draft_tokens,
            profile_seq_lens=profile_seq_lens,
        )
        for _ in range(_DCUT_PROFILE_WARMUPS):
            self._runner._dummy_run(target_tokens, **dummy_run_kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_DCUT_PROFILE_STEPS):
            self._runner._dummy_run(target_tokens, **dummy_run_kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / _DCUT_PROFILE_STEPS

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, next_power_of_2(max_tokens_per_req))
        num_blocks = (max_tokens_per_req + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        # Skip num_rejected_tokens (GPU-only); overestimating is fine here.
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=self.dflash_causal,
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
        num_query_tokens: int | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_query_tokens or num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Run the KV projection (GEMM + norms + RoPE) for memory profiling,
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context

        # Pre-insert context KVs directly into cache
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        if not self.dflash_causal:
            # Require all layers to support non-causal attention when required by DFlash
            for layer_name, attn_metadata in per_layer.items():
                assert getattr(attn_metadata, "causal", None) is False, (
                    f"Attention metadata for layer {layer_name} does not have"
                    " non-causal support, which is required for DFlash."
                    " Consider using a different attention backend, e.g FlashAttention."
                )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        return self.dflash_config.get("use_aux_hidden_state", True)

    @property
    def dflash_config(self):
        return getattr(self.draft_model_config.hf_config, "dflash_config", None) or {}
