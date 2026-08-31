# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec


class ShortConvAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]:
        return ShortConvAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata):
    pass


class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    metadata_cls = ShortConvAttentionMetadata


@dataclass
class PleShortConvAttentionMetadata(ShortConvAttentionMetadata):
    # Number of speculative-decode (multi-query / MTP) requests and the total
    # number of tokens they contribute. These are 0 when spec-decode is off.
    num_spec_decodes: int = 0
    num_spec_decode_tokens: int = 0
    num_actual_tokens: int = 0

    # Max query length among spec-decode requests (== num_speculative_tokens + 1).
    # Used as ``max_query_len`` for the varlen spec causal_conv1d_update.
    spec_query_len: int = 1

    # Max query length among the non-spec *prefill* requests, precomputed
    # CPU-side in the builder. The dilated PLE short-conv uses it to size its
    # packing buffer without a device->host sync (``lengths.max().item()``).
    # 0 when there are no prefill requests.
    max_prefill_query_len: int = 0
    query_start_loc: torch.Tensor | None = None

    # ``state_indices_tensor`` keeps the historical (non-spec) layout used by
    # all existing short-conv consumers: the conv-state slot for each regular
    # decode followed by each prefill request. When spec-decode is active this
    # only covers the non-spec requests.
    state_indices_tensor: torch.Tensor | None = None
    has_initial_states_d: torch.Tensor | None = None

    # ``non_spec_query_start_loc`` is the varlen cumulative token offset over
    # the non-spec requests only (decodes then prefills). It equals
    # ``query_start_loc`` when there are no spec-decode requests.
    non_spec_query_start_loc: torch.Tensor | None = None

    # Speculative-decode (MTP) conv metadata. Only column 0 of the block table
    # is needed for the convolution state, so these tensors are 1-D over the
    # spec-decode requests.
    spec_query_start_loc: torch.Tensor | None = None  # [num_spec_decodes + 1]
    spec_state_indices_tensor: torch.Tensor | None = None  # [num_spec_decodes]
    spec_sequence_masks: torch.Tensor | None = None  # [batch]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None


class PleShortConvAttentionBackend(ShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "PLE_SHORT_CONV_ATTN"

    @staticmethod
    def get_builder_cls() -> type["PleShortConvAttentionMetadataBuilder"]:
        return PleShortConvAttentionMetadataBuilder


class PleShortConvAttentionMetadataBuilder(ShortConvAttentionMetadataBuilder):
    metadata_cls = PleShortConvAttentionMetadata
    # Spec-decode requires a uniform (multi-token) decode batch for full
    # CUDA graph capture, matching the GDN backend.
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1
    supports_update_block_table = False

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.num_spec = self.num_spec_tokens
        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_capture_size = self.compilation_config.max_cudagraph_capture_size
        self.decode_cudagraph_max_bs = max_num_seqs
        self.decode_cudagraph_max_tokens = max_num_seqs * (self.num_spec + 1)
        if max_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs, max_capture_size
            )
            self.decode_cudagraph_max_tokens = min(
                self.decode_cudagraph_max_tokens, max_capture_size
            )

        # Persistent buffers reused during full CUDA graph capture and replay.
        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.bool, device=device
        )
        self.spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_tokens,), dtype=torch.int32, device=device
        )
        self.non_spec_token_indx = torch.empty(
            (self.decode_cudagraph_max_tokens,), dtype=torch.int32, device=device
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1,), dtype=torch.int32, device=device
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
        )
        self.has_initial_states_d = torch.empty(
            (self.decode_cudagraph_max_bs,), dtype=torch.bool, device=device
        )

    def _build_non_spec_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
        **kwargs: Any,
    ) -> PleShortConvAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            num_accepted_tokens=None,
            **kwargs,
        )
        assert isinstance(metadata, PleShortConvAttentionMetadata)

        state_indices_d = metadata.state_indices_tensor_d
        if state_indices_d is not None and state_indices_d.dim() > 1:
            state_indices_d = state_indices_d[:, 0]
        state_indices_p = metadata.state_indices_tensor_p
        if metadata.num_prefills == 0:
            assert state_indices_d is not None
            # BaseMambaAttentionMetadataBuilder pads decode state indices into
            # a persistent tensor for full CUDA graphs. Keep those rows so the
            # PLE decode receives one cache slot per graph-padded token.
            state_indices_tensor = state_indices_d
        elif metadata.num_decodes == 0:
            assert state_indices_p is not None
            state_indices_tensor = state_indices_p[: metadata.num_prefills]
        else:
            assert state_indices_d is not None
            assert state_indices_p is not None
            state_indices_tensor = torch.cat(
                (state_indices_d, state_indices_p[: metadata.num_prefills])
            )

        has_initial_states_d = None
        if metadata.num_decodes > 0:
            num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
            has_initial_states_d = num_computed_tokens[: metadata.num_decodes] > 0
            if (
                self.use_full_cuda_graph
                and metadata.num_prefills == 0
                and metadata.num_decodes <= self.decode_cudagraph_max_bs
            ):
                assert state_indices_d is not None
                # Prepare tensors for CUDA graph replay. Padded rows have no
                # initial state and use NULL_BLOCK_ID in state_indices_d.
                num_decode_rows = state_indices_d.numel()
                self.has_initial_states_d[: metadata.num_decodes].copy_(
                    has_initial_states_d, non_blocking=True
                )
                self.has_initial_states_d[metadata.num_decodes : num_decode_rows].fill_(
                    False
                )
                has_initial_states_d = self.has_initial_states_d[:num_decode_rows]

        max_prefill_query_len = 0
        if metadata.num_prefills > 0:
            query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
            max_prefill_query_len = int(
                query_lens_cpu[
                    metadata.num_decodes : (
                        metadata.num_decodes + metadata.num_prefills
                    )
                ]
                .max()
                .item()
            )

        return replace(
            metadata,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            spec_query_len=self.num_spec + 1,
            max_prefill_query_len=max_prefill_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states_d=has_initial_states_d,
            non_spec_query_start_loc=common_attn_metadata.query_start_loc,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> PleShortConvAttentionMetadata:
        m = common_attn_metadata
        spec_sequence_masks_cpu: torch.Tensor | None = None
        # Detect speculative-decode requests. We use -1 to mark prefill and
        # plain-decode requests, so any value >= 0 is a (multi-query)
        # spec-decode request.
        if self.use_spec_decode and num_decode_draft_tokens_cpu is not None:
            candidate_mask = num_decode_draft_tokens_cpu[: m.num_reqs] >= 0
            if bool(candidate_mask.any().item()):
                spec_sequence_masks_cpu = candidate_mask

        if spec_sequence_masks_cpu is None:
            return self._build_non_spec_metadata(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                num_decode_draft_tokens_cpu,
                **kwargs,
            )

        del common_prefix_len, fast_build, kwargs
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        query_lens_cpu = torch.diff(query_start_loc_cpu)
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        if query_start_loc.device.type == "cpu":
            spec_sequence_masks = spec_sequence_masks_cpu
        else:
            spec_sequence_masks = async_tensor_h2d(
                spec_sequence_masks_cpu, device=query_start_loc.device
            )

        # For causal_conv1d (non-spec prefill Triton kernel metadata).
        nums_dict = None
        batch_ptr = None
        token_chunk_offset_ptr = None
        has_initial_states_p = None
        has_initial_states_d = None
        num_computed_tokens_p = None
        # Original request indices of the non-spec requests, ordered
        # [decodes, prefills]. Used to gather per-request data consistently.
        non_spec_req_idx_cpu: torch.Tensor | None = None

        query_lens = torch.diff(query_start_loc)
        # Per-request classification by mask, NOT by position. With
        # spec-decode, the front decode group can contain both spec-decode
        # requests and plain non-spec single-token decodes. Spec requests are
        # therefore not guaranteed to occupy the first num_spec_decodes slots.
        non_spec_mask_cpu = ~spec_sequence_masks_cpu
        decode_mask_cpu = non_spec_mask_cpu & (query_lens_cpu == 1)
        prefill_mask_cpu = non_spec_mask_cpu & (query_lens_cpu > 1)

        num_spec_decodes = int(spec_sequence_masks_cpu.sum().item())
        num_decodes = int(decode_mask_cpu.sum().item())
        num_prefills = int(prefill_mask_cpu.sum().item())
        num_decode_tokens = num_decodes
        num_prefill_tokens = int(query_lens_cpu[prefill_mask_cpu].sum().item())
        num_spec_decode_tokens = int(
            query_lens_cpu[spec_sequence_masks_cpu].sum().item()
        )
        # Max prefill query length, CPU-side (no device-to-host sync) for the
        # PLE dilated short-conv packing buffer.
        max_prefill_query_len = (
            int(query_lens_cpu[prefill_mask_cpu].max().item())
            if num_prefills > 0
            else 0
        )

        # Original request indices grouped as
        # [spec | non-spec decode | non-spec prefill]; each group keeps the
        # original (already reordered) relative order via a stable nonzero.
        spec_req_idx_cpu = spec_sequence_masks_cpu.nonzero(as_tuple=True)[0]
        decode_req_idx_cpu = decode_mask_cpu.nonzero(as_tuple=True)[0]
        prefill_req_idx_cpu = prefill_mask_cpu.nonzero(as_tuple=True)[0]
        non_spec_req_idx_cpu = torch.cat((decode_req_idx_cpu, prefill_req_idx_cpu))
        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)
        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)

        if num_decodes == 0 and num_prefills == 0:
            # Pure speculative-decode batch: all real tokens are spec tokens.
            spec_token_indx = torch.arange(
                num_spec_decode_tokens,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = None
            spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
            non_spec_query_start_loc = None
            non_spec_query_start_loc_cpu = None
        else:
            # Mixed batch: build a per-token group key consistent with the
            # request grouping above (spec=0 | decode=1 | prefill=2) and a
            # stable sort, so tokens of each request stay contiguous and in
            # request order. This yields spec tokens first, then the non-spec
            # [decode, prefill] tokens.
            req_group = torch.full(
                (m.num_reqs,),
                2,
                dtype=torch.int64,
                device=query_start_loc.device,
            )
            req_group[spec_req_idx] = 0
            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1
            token_group = torch.repeat_interleave(req_group, query_lens)
            token_perm = torch.argsort(token_group, stable=True)
            spec_token_indx = token_perm[:num_spec_decode_tokens]
            non_spec_token_indx = token_perm[num_spec_decode_tokens:]

            spec_state_indices_tensor = block_table_tensor[spec_req_idx, 0]
            non_spec_state_indices_tensor = block_table_tensor[non_spec_req_idx, 0]
            spec_query_start_loc = torch.zeros(
                num_spec_decodes + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(query_lens[spec_req_idx], dim=0, out=spec_query_start_loc[1:])
            non_spec_query_start_loc = torch.zeros(
                num_decodes + num_prefills + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            torch.cumsum(
                query_lens[non_spec_req_idx],
                dim=0,
                out=non_spec_query_start_loc[1:],
            )
            non_spec_query_start_loc_cpu = torch.zeros(
                num_decodes + num_prefills + 1, dtype=torch.int32
            )
            torch.cumsum(
                query_lens_cpu[non_spec_req_idx_cpu],
                dim=0,
                out=non_spec_query_start_loc_cpu[1:],
            )

        assert num_accepted_tokens is not None
        # Accepted-token counts must follow the same request order as the
        # speculative state indices.
        num_accepted_tokens = num_accepted_tokens[
            spec_req_idx_cpu.to(num_accepted_tokens.device)
        ]

        # Compute the conv-state slots for the non-spec decode/prefill split,
        # plus the initial-state masks and Triton causal_conv1d metadata.
        if non_spec_state_indices_tensor is None:
            state_indices_tensor = block_table_tensor[:0, 0]
        else:
            state_indices_tensor = non_spec_state_indices_tensor

        # Build the regular decode/prefill state metadata inherited from the
        # generic short-conv metadata contract.
        query_start_loc_p = None
        query_start_loc_d = None
        state_indices_tensor_p = None
        state_indices_tensor_d = None
        if num_decodes > 0 or num_prefills > 0:
            num_computed_tokens = m.compute_num_computed_tokens()
            if non_spec_req_idx_cpu is not None:
                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)
                num_computed_tokens = num_computed_tokens[non_spec_req_idx]

            state_indices_tensor_d = state_indices_tensor[:num_decodes]
            state_indices_tensor_p = state_indices_tensor[
                num_decodes : num_decodes + num_prefills
            ]
            if num_decodes > 0:
                has_initial_states_d = num_computed_tokens[:num_decodes] > 0
                assert non_spec_query_start_loc is not None
                query_start_loc_d = non_spec_query_start_loc[: num_decodes + 1]
            if num_prefills > 0:
                num_computed_tokens_p = num_computed_tokens[
                    num_decodes : num_decodes + num_prefills
                ]
                has_initial_states_p = num_computed_tokens_p > 0
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                query_start_loc_p = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                query_start_loc_p_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                if query_start_loc.device.type != "cpu":
                    nums_dict, batch_ptr, token_chunk_offset_ptr = (
                        compute_causal_conv1d_metadata(
                            query_start_loc_p_cpu,
                            device=query_start_loc.device,
                        )
                    )

        # Prepare persistent tensors for CUDA graph capture and replay.
        # ``m.num_actual_tokens`` is already padded by the model runner.
        # Request-level buffers use ``m.num_reqs`` while token-level buffers
        # use their independently bounded token count.
        batch_size = m.num_reqs
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and spec_sequence_masks is not None
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_tokens
        ):
            assert spec_state_indices_tensor is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:batch_size].copy_(
                spec_sequence_masks[:batch_size], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]

            assert spec_query_start_loc is not None
            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            assert num_accepted_tokens is not None
            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        return PleShortConvAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_reqs=m.num_reqs,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            spec_query_len=self.num_spec + 1,
            max_prefill_query_len=max_prefill_query_len,
            query_start_loc=query_start_loc,
            state_indices_tensor=state_indices_tensor,
            has_initial_states_p=has_initial_states_p,
            has_initial_states_d=has_initial_states_d,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_query_start_loc=spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            query_start_loc_p=query_start_loc_p,
            query_start_loc_d=query_start_loc_d,
            state_indices_tensor_p=state_indices_tensor_p,
            state_indices_tensor_d=state_indices_tensor_d,
            num_computed_tokens_p=num_computed_tokens_p,
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
            seq_lens=m.seq_lens,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> PleShortConvAttentionMetadata:
        """Build metadata for full CUDA graph capture.

        Currently, only decode is supported for full CUDA graphs with
        short-conv.
        """
        m = common_attn_metadata
        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_tokens
        ), (
            "ShortConv only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture size ({self.decode_cudagraph_max_bs}) and "
            f"number of tokens ({m.num_actual_tokens}) <= "
            f"token capture size ({self.decode_cudagraph_max_tokens})."
        )

        if self.use_spec_decode:
            num_accepted_tokens = torch.diff(m.query_start_loc)
            num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
            return self.build(
                0,
                m,
                num_accepted_tokens=num_accepted_tokens,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            )
        return self.build(0, m)
