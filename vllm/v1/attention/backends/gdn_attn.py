# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # Read-side physical block ids for the mamba "align" SSM src redirect. When
    # set, the SSM kernels source their initial state from these blocks instead
    # of [non_]spec_state_indices_tensor, eliminating the SSM temporal pre-copy.
    spec_src_state_indices: torch.Tensor | None = None  # shape: [num_spec_decodes,]
    non_spec_src_state_indices: torch.Tensor | None = None  # shape: [num_decodes,]

    # Read-side CONV src: physical block id (prev running block) + pre-reset
    # intra-block token offset, so the conv kernel reads its init conv window
    # from the prev block at that offset (copy-free conv pre).
    spec_conv_src_state_indices: torch.Tensor | None = None
    non_spec_conv_src_state_indices: torch.Tensor | None = None
    spec_conv_src_offset: torch.Tensor | None = None
    non_spec_conv_src_offset: torch.Tensor | None = None

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal["triton", "flashinfer", "cutedsl"]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        # Cache the cache-mode flag once; per-step paths just read this.
        self.is_mamba_cache_align: bool = (
            vllm_config.cache_config.mamba_cache_mode == "align"
        )

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        # Persistent buffers for the SSM src physical block ids (mamba "align").
        # Required for full CUDAGraph: the SSM kernel's HAS_SRC_INDICES is a
        # triton constexpr baked at CAPTURE; the src must be routed through these
        # fixed-address buffers so capture bakes HAS_SRC=True (NULL-filled) and
        # each replay step copies the real src in. Only needed in align mode.
        self.spec_src_state_indices_buf: torch.Tensor | None = None
        self.non_spec_src_state_indices_buf: torch.Tensor | None = None
        if self.is_mamba_cache_align:
            self.spec_src_state_indices_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.non_spec_src_state_indices_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            # CONV src physical-block + offset persistent buffers (same purpose).
            self.spec_conv_src_state_indices_buf = torch.empty(
                (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
            )
            self.non_spec_conv_src_state_indices_buf = torch.empty(
                (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
            )
            self.spec_conv_src_offset_buf = torch.empty(
                (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
            )
            self.non_spec_conv_src_offset_buf = torch.empty(
                (self.decode_cudagraph_max_bs,), dtype=torch.int32, device=device
            )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        align_src_state_indices: torch.Tensor | None = None,
        align_conv_src_state_indices: torch.Tensor | None = None,
        align_conv_src_offset: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        # Resolve the SSM src COLUMN (batch order; -1 == fresh) to a physical
        # block id against the FULL block table. The previous running block can
        # lie OUTSIDE the windowed `block_table_tensor` on a block-boundary
        # crossing, so we index m.block_table_tensor directly here. Split per
        # spec/non-spec below, mirroring the state-index tensors.
        align_src_full: torch.Tensor | None = None
        if align_src_state_indices is not None and self.is_mamba_cache_align:
            num_reqs_full = m.block_table_tensor.size(0)
            col = align_src_state_indices[:num_reqs_full].to(torch.int64)
            max_col = m.block_table_tensor.size(1) - 1
            col_safe = col.clamp(min=0, max=max_col)
            rows = torch.arange(
                num_reqs_full,
                device=m.block_table_tensor.device,
                dtype=torch.int64,
            )
            phys = m.block_table_tensor[rows, col_safe].to(torch.int32)
            # col < 0 == fresh -> 0 (NULL_BLOCK_ID) -> kernel starts from zero.
            align_src_full = torch.where(col >= 0, phys, torch.zeros_like(phys))

        # Same resolution for the CONV src column (= state_idx, the prev MAIN
        # block). The per-seq offset is carried through unchanged (batch order).
        align_conv_src_full: torch.Tensor | None = None
        align_conv_off_full: torch.Tensor | None = None
        if align_conv_src_state_indices is not None and self.is_mamba_cache_align:
            num_reqs_full = m.block_table_tensor.size(0)
            ccol = align_conv_src_state_indices[:num_reqs_full].to(torch.int64)
            max_col = m.block_table_tensor.size(1) - 1
            ccol_safe = ccol.clamp(min=0, max=max_col)
            crows = torch.arange(
                num_reqs_full, device=m.block_table_tensor.device, dtype=torch.int64
            )
            cphys = m.block_table_tensor[crows, ccol_safe].to(torch.int32)
            align_conv_src_full = torch.where(
                ccol >= 0, cphys, torch.zeros_like(cphys)
            )
            align_conv_off_full = align_conv_src_offset[:num_reqs_full].to(torch.int32)

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
            # Pure non-spec batch: every row is a non-spec decode, so the
            # resolved physical src blocks are already in non-spec order.
            spec_src_state_indices = None
            non_spec_src_state_indices = (
                align_src_full.contiguous() if align_src_full is not None else None
            )
            spec_conv_src_state_indices = None
            spec_conv_src_offset = None
            non_spec_conv_src_state_indices = (
                align_conv_src_full.contiguous()
                if align_conv_src_full is not None
                else None
            )
            non_spec_conv_src_offset = (
                align_conv_off_full.contiguous()
                if align_conv_off_full is not None
                else None
            )
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # num_decodes and num_spec_decodes are mutually exclusive.
            # Reclassify non-spec decodes as prefills when spec decodes
            # exist — the prefill kernel handles 1-token sequences with
            # initial state correctly, producing identical results.
            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                if align_src_full is not None:
                    spec_src_state_indices = align_src_full[
                        spec_sequence_masks_cpu
                    ].contiguous()
                else:
                    spec_src_state_indices = None
                non_spec_src_state_indices = None
                if align_conv_src_full is not None:
                    spec_conv_src_state_indices = align_conv_src_full[
                        spec_sequence_masks_cpu
                    ].contiguous()
                    spec_conv_src_offset = align_conv_off_full[
                        spec_sequence_masks_cpu
                    ].contiguous()
                else:
                    spec_conv_src_state_indices = None
                    spec_conv_src_offset = None
                non_spec_conv_src_state_indices = None
                non_spec_conv_src_offset = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks_cpu, 0
                ]
                if align_src_full is not None:
                    spec_src_state_indices = align_src_full[
                        spec_sequence_masks_cpu
                    ].contiguous()
                    non_spec_src_state_indices = align_src_full[
                        ~spec_sequence_masks_cpu
                    ].contiguous()
                else:
                    spec_src_state_indices = None
                    non_spec_src_state_indices = None
                if align_conv_src_full is not None:
                    spec_conv_src_state_indices = align_conv_src_full[
                        spec_sequence_masks_cpu
                    ].contiguous()
                    spec_conv_src_offset = align_conv_off_full[
                        spec_sequence_masks_cpu
                    ].contiguous()
                    non_spec_conv_src_state_indices = align_conv_src_full[
                        ~spec_sequence_masks_cpu
                    ].contiguous()
                    non_spec_conv_src_offset = align_conv_off_full[
                        ~spec_sequence_masks_cpu
                    ].contiguous()
                else:
                    spec_conv_src_state_indices = None
                    spec_conv_src_offset = None
                    non_spec_conv_src_state_indices = None
                    non_spec_conv_src_offset = None

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks_cpu],
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        if num_prefills > 0:
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            if self.gdn_prefill_backend == "cutedsl":
                from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                    prepare_metadata_cutedsl,
                )

                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                total_tokens = int(non_spec_query_start_loc_cpu[-1].item())
                chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
                    non_spec_query_start_loc,
                    total_tokens,
                    FLA_CHUNK_SIZE,
                )
            else:
                gpu_device = query_start_loc.device
                # Only prefill batches use FLA chunk ops.
                # Pre-compute on CPU and async-copy to GPU to avoid
                # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
                from vllm.model_executor.layers.fla.ops.index import (
                    prepare_chunk_indices,
                    prepare_chunk_offsets,
                )

                assert non_spec_query_start_loc_cpu is not None
                chunk_indices = prepare_chunk_indices(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)
                chunk_offsets = prepare_chunk_offsets(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

            # SSM src (align): route through the persistent buffer so the
            # captured kernel bakes HAS_SRC_INDICES=True. At capture
            # spec_src_state_indices is None -> NULL-fill (kernel skips, safe);
            # at replay copy the real src in. Tail padded with NULL_BLOCK_ID.
            if self.is_mamba_cache_align:
                assert self.spec_src_state_indices_buf is not None
                if spec_src_state_indices is not None:
                    self.spec_src_state_indices_buf[:num_spec_decodes].copy_(
                        spec_src_state_indices, non_blocking=True
                    )
                else:
                    self.spec_src_state_indices_buf[:num_spec_decodes].fill_(
                        NULL_BLOCK_ID
                    )
                spec_src_state_indices = self.spec_src_state_indices_buf[:batch_size]
                spec_src_state_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)

                # CONV src + offset persistent buffers (same capture/replay rule).
                if spec_conv_src_state_indices is not None:
                    self.spec_conv_src_state_indices_buf[:num_spec_decodes].copy_(
                        spec_conv_src_state_indices, non_blocking=True
                    )
                    self.spec_conv_src_offset_buf[:num_spec_decodes].copy_(
                        spec_conv_src_offset, non_blocking=True
                    )
                else:
                    self.spec_conv_src_state_indices_buf[:num_spec_decodes].fill_(
                        NULL_BLOCK_ID
                    )
                    self.spec_conv_src_offset_buf[:num_spec_decodes].fill_(0)
                spec_conv_src_state_indices = self.spec_conv_src_state_indices_buf[
                    :batch_size
                ]
                spec_conv_src_state_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)
                spec_conv_src_offset = self.spec_conv_src_offset_buf[:batch_size]
                spec_conv_src_offset[num_spec_decodes:].fill_(0)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

            # SSM src (align): route through the persistent buffer so the
            # captured kernel bakes HAS_SRC_INDICES=True (see spec branch).
            if self.is_mamba_cache_align:
                assert self.non_spec_src_state_indices_buf is not None
                if non_spec_src_state_indices is not None:
                    self.non_spec_src_state_indices_buf[:num_decodes].copy_(
                        non_spec_src_state_indices, non_blocking=True
                    )
                else:
                    self.non_spec_src_state_indices_buf[:num_decodes].fill_(
                        NULL_BLOCK_ID
                    )
                non_spec_src_state_indices = self.non_spec_src_state_indices_buf[
                    :batch_size
                ]
                non_spec_src_state_indices[num_decodes:].fill_(NULL_BLOCK_ID)

                if non_spec_conv_src_state_indices is not None:
                    self.non_spec_conv_src_state_indices_buf[:num_decodes].copy_(
                        non_spec_conv_src_state_indices, non_blocking=True
                    )
                    self.non_spec_conv_src_offset_buf[:num_decodes].copy_(
                        non_spec_conv_src_offset, non_blocking=True
                    )
                else:
                    self.non_spec_conv_src_state_indices_buf[:num_decodes].fill_(
                        NULL_BLOCK_ID
                    )
                    self.non_spec_conv_src_offset_buf[:num_decodes].fill_(0)
                non_spec_conv_src_state_indices = (
                    self.non_spec_conv_src_state_indices_buf[:batch_size]
                )
                non_spec_conv_src_state_indices[num_decodes:].fill_(NULL_BLOCK_ID)
                non_spec_conv_src_offset = self.non_spec_conv_src_offset_buf[
                    :batch_size
                ]
                non_spec_conv_src_offset[num_decodes:].fill_(0)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            spec_src_state_indices=spec_src_state_indices,
            non_spec_src_state_indices=non_spec_src_state_indices,
            spec_conv_src_state_indices=spec_conv_src_state_indices,
            non_spec_conv_src_state_indices=non_spec_conv_src_state_indices,
            spec_conv_src_offset=spec_conv_src_offset,
            non_spec_conv_src_offset=non_spec_conv_src_offset,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
