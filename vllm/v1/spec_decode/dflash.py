# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import Any

import torch
from typing_extensions import override

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

logger = init_logger(__name__)


@dataclass
class _DFlashGraphBucket:
    # Bucket shape info
    num_context_tokens: int
    total_query_tokens: int
    num_kv_tokens: int
    num_input_tokens: int

    # Static tensors used by captured graph
    input_ids: torch.Tensor
    positions: torch.Tensor
    hidden_states: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    num_tokens_across_dp: torch.Tensor | None

    # Metadata / context
    per_layer_attn_metadata: dict[str, Any]

    # Graph output and graph object
    output_hidden_states: torch.Tensor
    graph: torch.cuda.CUDAGraph


class DFlashModelProposer(SpecDecodeBaseProposer):
    """DFlash draft model proposer.

    - Uses target hidden states as KV; context length can vary.
    - Parallel drafting for queries (sampled token + several MASK tokens).
    - Per-shape CUDA graph bucketing strategy (aligned with dflash 0.13).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # Same as EagleProposer: pass target hidden_states to draft model
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Read mask_token_id from draft model config.json dflash_config
        draft_hf_config = self.draft_model_config.hf_config
        dflash_cfg = getattr(draft_hf_config, "dflash_config", {})
        if "mask_token_id" in dflash_cfg:
            self.mask_token_id = dflash_cfg["mask_token_id"]
        else:
            raise ValueError(
                "DFlash draft model config must have "
                "`dflash_config.mask_token_id` in its config.json."
            )

        # CUDA graph control
        self.compilation_config = self.vllm_config.compilation_config
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if cudagraph_mode != CUDAGraphMode.NONE and not cudagraph_mode.has_mode(
            CUDAGraphMode.PIECEWISE
        ):
            logger.warning(
                "DFlash draft model only supports PIECEWISE CUDA graphs. "
                "Please set compilation_config.cudagraph_mode to PIECEWISE "
                "or FULL_AND_PIECEWISE to enable CUDA graphs for DFlash."
            )

        self.use_cuda_graph: bool = (
            cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE)
            and not self.speculative_config.enforce_eager
        )
        # One bucket per KV shape
        self._dflash_graph_pool = torch.cuda.graph_pool_handle()
        self._dflash_graph_buckets: dict[tuple[int, int, int], _DFlashGraphBucket] = {}

    # key: (batch_size, num_context_tokens, total_query_tokens)

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        (_, num_input_tokens, num_tokens_across_dp) = (
            self._determine_batch_execution_and_padding(
                num_tokens, use_cudagraphs=False
            )
        )
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=slot_mappings or {},
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(2 * num_input_tokens),
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=None,
            )            
           
    @override
    def _get_slot_mapping(
        self,
        num_tokens_or_slot_mapping,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """DFlash constructs complete slot_mappings externally,
        so just wrap the tensor in a per-layer dict."""
        if (
            isinstance(num_tokens_or_slot_mapping, torch.Tensor)
            and slot_mapping is None
        ):
            sm = num_tokens_or_slot_mapping
        else:
            sm = (
                slot_mapping if slot_mapping is not None else num_tokens_or_slot_mapping
            )
        return {name: sm for name in self._draft_attn_layer_names}

    # ---------------- DFlash-specific internal utilities ----------------

    def _build_dflash_common_attn_metadata(
        self,
        *,
        common_attn_metadata: CommonAttentionMetadata,
        position_ids: torch.Tensor,
        num_query_tokens: int,
        slot_mapping: torch.Tensor,
    ) -> CommonAttentionMetadata:
        """Build non-causal metadata for DFlash 
        from original CommonAttentionMetadata."""
        batch_size = common_attn_metadata.num_reqs

        # For batch_size=1, query_start_loc is always [0, num_query_tokens]
        query_start_loc = self.arange[: batch_size + 1] * num_query_tokens
        query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
            * num_query_tokens
        )

        # seq_lens adds query length on top of original (kept dynamic)
        seq_lens = common_attn_metadata.seq_lens + num_query_tokens

        max_seq_len = self.max_model_len

        return replace(
            common_attn_metadata,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=None,
            num_actual_tokens=batch_size * num_query_tokens,
            max_query_len=num_query_tokens,
            max_seq_len=max_seq_len,
            slot_mapping=slot_mapping,
            causal=False,
        )

    def _run_dflash_eager(
        self,
        *,
        total_query_tokens: int,
        num_kv_tokens: int,
        num_context_tokens: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        per_layer_attn_metadata: dict[str, Any],
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Run DFlash draft model forward in eager mode."""
        (_, num_query_tokens_dp_padded, num_tokens_across_dp) = (
            self._determine_batch_execution_and_padding(              
                total_query_tokens, use_cudagraphs=False
            )
        )
        self._set_positions(num_kv_tokens, position_ids)
        self.input_ids[:total_query_tokens] = input_ids[:total_query_tokens]
        if num_query_tokens_dp_padded > total_query_tokens:
            self.input_ids[total_query_tokens:num_query_tokens_dp_padded].fill_(0)

        self.hidden_states[:num_context_tokens] = target_hidden_states

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_query_tokens_dp_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=self._get_slot_mapping(slot_mapping),
        ):
            ret_hidden_states = self.model(
                input_ids=self.input_ids[:num_query_tokens_dp_padded],
                positions=self._get_positions(num_kv_tokens),
                hidden_states=self.hidden_states[:num_context_tokens],
                inputs_embeds=None,
            )
        return ret_hidden_states

    def _capture_dflash_bucket(
        self,
        *,
        bucket_key: tuple[int, int, int],
        total_query_tokens: int,
        num_query_tokens_per_req: int,
        num_context_tokens: int,
        num_kv_tokens: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> _DFlashGraphBucket:
        """Capture a CUDA graph bucket for the given KV shape."""
        _, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                total_query_tokens, use_cudagraphs=False
            )
        )
        if num_tokens_across_dp is not None:
            num_tokens_across_dp = num_tokens_across_dp.clone()
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        input_ids_static = torch.empty(
            (num_input_tokens,), dtype=self.input_ids.dtype, device=self.device
        )
        if self.uses_mrope:
            positions_static = torch.empty(
                (3, num_kv_tokens), dtype=position_ids.dtype, device=self.device
            )
        else:
            positions_static = torch.empty(
                (num_kv_tokens,), dtype=position_ids.dtype, device=self.device
            )

        hidden_states_static = torch.empty(
            (num_context_tokens, self.hidden_size), dtype=self.dtype, device=self.device
        )
        slot_mapping_static = torch.empty_like(slot_mapping)
        seq_lens_static = common_attn_metadata.seq_lens.clone()

        common_attn_static = self._build_dflash_common_attn_metadata(
            common_attn_metadata=common_attn_metadata,
            position_ids=position_ids,
            num_query_tokens=num_query_tokens_per_req,
            slot_mapping=slot_mapping_static,
        )
        common_attn_static = replace(common_attn_static, seq_lens=seq_lens_static)

        per_layer_attn_metadata: dict[str, Any] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_static, draft_index=0
            )
            if hasattr(attn_metadata, "causal"):
                assert not attn_metadata.causal, (
                    "DFlash proposer requires non-causal attention. "
                    "Choose a different attention backend."
                )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        # Initialize static buffer contents
        input_ids_static[:total_query_tokens].copy_(input_ids[:total_query_tokens])
        if num_input_tokens > total_query_tokens:
            input_ids_static[total_query_tokens:].fill_(0)

        positions_static.copy_(position_ids)
        hidden_states_static.copy_(target_hidden_states)
        slot_mapping_static.copy_(slot_mapping)
        seq_lens_static.copy_(common_attn_static.seq_lens)

        graph = torch.cuda.CUDAGraph()

        # warmup
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=self._get_slot_mapping(slot_mapping_static),
        ):
            output_hidden_states = self.model(
                input_ids=input_ids_static,
                positions=positions_static,
                hidden_states=hidden_states_static,
                inputs_embeds=None,
            )

        # capture
        with (
            torch.cuda.graph(
                graph,
                self._dflash_graph_pool,
            ),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=self._get_slot_mapping(slot_mapping_static),
            ),
        ):
            output_hidden_states = self.model(
                input_ids=input_ids_static,
                positions=positions_static,
                hidden_states=hidden_states_static,
                inputs_embeds=None,
            )

        bucket = _DFlashGraphBucket(
            num_context_tokens=num_context_tokens,
            total_query_tokens=total_query_tokens,
            num_kv_tokens=num_kv_tokens,
            num_input_tokens=num_input_tokens,
            input_ids=input_ids_static,
            positions=positions_static,
            hidden_states=hidden_states_static,
            slot_mapping=slot_mapping_static,
            seq_lens=seq_lens_static,
            num_tokens_across_dp=num_tokens_across_dp,
            per_layer_attn_metadata=per_layer_attn_metadata,
            output_hidden_states=output_hidden_states,
            graph=graph,
        )
        self._dflash_graph_buckets[bucket_key] = bucket
        return bucket

    def _run_dflash_graph(
        self,
        *,
        bucket: _DFlashGraphBucket,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured DFlash graph."""
        bucket.input_ids[: bucket.total_query_tokens].copy_(
            input_ids[: bucket.total_query_tokens]
        )
        if bucket.num_input_tokens > bucket.total_query_tokens:
            bucket.input_ids[bucket.total_query_tokens : bucket.num_input_tokens].fill_(
                0
            )
        bucket.positions.copy_(position_ids)
        bucket.hidden_states.copy_(target_hidden_states)
        bucket.slot_mapping.copy_(slot_mapping)
        bucket.seq_lens.copy_(seq_lens)

        bucket.graph.replay()
        return bucket.output_hidden_states

    def _propose_dflash_core(
        self,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.num_reqs
        device = self.device

        num_query_tokens = 1 + self.num_speculative_tokens
        total_query_tokens = batch_size * num_query_tokens
        num_context_tokens = target_hidden_states.shape[0]
        num_kv_tokens = num_context_tokens + total_query_tokens

        assert self.runner is not None

        # target_hidden_states: [num_tokens, hidden_size]
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size

        if (
            self.uses_xdrope_dim > 0
            and self.draft_uses_xdrope_dim == 0
            and target_positions.ndim == 2
        ):
            target_positions = target_positions[0]

        # Use 1D base positions for per-request last position and slot computation;
        # position_ids fed to the model may be 1D or 2D (M-RoPE).
        if target_positions.ndim == 2:
            # [3, num_context_tokens]
            base_target_positions = target_positions[0]
        else:
            # [num_context_tokens]
            base_target_positions = target_positions

        # End index (exclusive) of each request's context
        # Diff of query_start_loc equals tokens per request
        query_start_loc = common_attn_metadata.query_start_loc.to(
            device=device, dtype=torch.long
        )
        ctx_lens = query_start_loc[1:] - query_start_loc[:-1]  # [batch_size]
        ctx_end_indices = query_start_loc[1:]  # [batch_size]

        # Position of each request's last context token
        last_positions = base_target_positions[ctx_end_indices - 1]  # [batch_size]

        # Per-request query positions: [last+1, last+2, ..., last+num_query_tokens]
        query_offsets = torch.arange(
            1,
            num_query_tokens + 1,
            device=device,
            dtype=base_target_positions.dtype,
        )
        query_positions = last_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
        query_positions_flat = query_positions.reshape(-1)  # [total_query_tokens]

        # Assemble position_ids (fed to draft model)
        if target_positions.ndim == 2:
            # M-RoPE: expand to [3, total_query_tokens]
            query_positions_mrope = query_positions_flat.unsqueeze(0).expand(
                target_positions.shape[0], -1
            )
            position_ids = torch.cat([target_positions, query_positions_mrope], dim=1)
            # Slot mapping uses 1D base positions
            flat_position_ids = position_ids[0]
            assert position_ids.shape[1] == num_kv_tokens
        else:
            position_ids = torch.cat([target_positions, query_positions_flat], dim=0)
            flat_position_ids = position_ids
            assert position_ids.shape[0] == num_kv_tokens

        # Build req_indices: which request each token belongs to
        ctx_req_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.long),
            ctx_lens.to(torch.long),
        )
        query_req_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device, dtype=torch.long),
            num_query_tokens,
        )
        req_indices = torch.cat([ctx_req_indices, query_req_indices], dim=0)
        assert req_indices.shape[0] == num_kv_tokens

        # Build slot_mapping
        block_size = self.block_size
        block_numbers = (flat_position_ids // block_size).to(torch.long)
        block_ids = common_attn_metadata.block_table_tensor[req_indices, block_numbers]
        slot_mapping = block_ids * block_size + (flat_position_ids % block_size)

        # Build DFlash non-causal metadata from original metadata
        dflash_common_attn = self._build_dflash_common_attn_metadata(
            common_attn_metadata=common_attn_metadata,
            position_ids=position_ids,
            num_query_tokens=num_query_tokens,
            slot_mapping=slot_mapping,
        )

        per_layer_attn_metadata: dict[str, Any] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=dflash_common_attn,
                draft_index=0,
            )
            if hasattr(attn_metadata, "causal"):
                assert not attn_metadata.causal, (
                    "DFlash proposer requires non-causal attention. "
                    "Choose a different attention backend."
                )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        self.input_ids[:total_query_tokens] = self.mask_token_id
        first_query_indices = torch.arange(
            0, total_query_tokens, num_query_tokens, device=device, dtype=torch.long
        )
        self.input_ids[first_query_indices] = next_token_ids.to(self.input_ids.dtype)
        dflash_input_ids = self.input_ids[:total_query_tokens]

        # CUDA graph strategy
        can_use_dflash_graph = (
            self.use_cuda_graph
            and total_query_tokens <= self.compilation_config.max_cudagraph_capture_size
            and num_context_tokens <= total_query_tokens
        )

        if not can_use_dflash_graph:
            ret_hidden_states = self._run_dflash_eager(
                total_query_tokens=total_query_tokens,
                num_kv_tokens=num_kv_tokens,
                num_context_tokens=num_context_tokens,
                input_ids=dflash_input_ids,
                position_ids=position_ids,
                target_hidden_states=target_hidden_states,
                per_layer_attn_metadata=per_layer_attn_metadata,
                slot_mapping=slot_mapping,
            )
        else:
            bucket_key = (batch_size, num_context_tokens, total_query_tokens)
            bucket = self._dflash_graph_buckets.get(bucket_key)
            if bucket is None:
                bucket = self._capture_dflash_bucket(
                    bucket_key=bucket_key,
                    total_query_tokens=total_query_tokens,
                    num_query_tokens_per_req=num_query_tokens,
                    num_context_tokens=num_context_tokens,
                    num_kv_tokens=num_kv_tokens,
                    input_ids=dflash_input_ids,
                    position_ids=position_ids,
                    target_hidden_states=target_hidden_states,
                    slot_mapping=slot_mapping,
                    common_attn_metadata=dflash_common_attn,
                )

            ret_hidden_states = self._run_dflash_graph(
                bucket=bucket,
                input_ids=dflash_input_ids,
                position_ids=position_ids,
                target_hidden_states=target_hidden_states,
                slot_mapping=slot_mapping,
                seq_lens=dflash_common_attn.seq_lens,
            )

        # ret_hidden_states: [total_query_tokens, hidden_size]
        # Per-request layout: [sampled, mask1, mask2, ...]
        ret_hidden_states = ret_hidden_states[:total_query_tokens]
        ret_hidden_states = ret_hidden_states.view(batch_size, num_query_tokens, -1)
        valid_hidden_states = ret_hidden_states[:, 1:, :].reshape(
            batch_size * self.num_speculative_tokens, -1
        )

        logits = self.model.compute_logits(valid_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        return draft_token_ids.view(batch_size, self.num_speculative_tokens)
