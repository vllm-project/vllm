# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
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


@triton.jit
def _gather_resolve_align_src_kernel(
    idx_mapping_ptr,  # [>=n] batch position -> req state slot (-1 = filtered/PP)
    src_ssm_col_ptr,  # [max_reqs] req-order SSM src column (-1 = fresh)
    conv_src_col_ptr,  # [max_reqs] req-order conv src column (-1 = fresh)
    conv_src_off_ptr,  # [max_reqs] req-order conv token offset
    block_table_ptr,  # [num_reqs, max_blocks] int32 (FULL, non-windowed)
    bt_stride0,
    bt_stride1,
    max_col,
    ssm_phys_ptr,  # [num_reqs] int32 out (NULL_BLOCK_ID=0 for fresh)
    conv_phys_ptr,  # [num_reqs] int32 out
    conv_off_ptr,  # [num_reqs] int32 out (batch order)
    n,
    num_reqs,
    BLOCK: tl.constexpr,
):
    """Gather the per-req align src COLUMNS into batch order (via idx_mapping)
    and resolve them to physical block ids in one launch (SSM + conv together).
    Rows in [n, num_reqs) are padding and rows whose req < 0 are filtered (PP);
    both resolve to fresh (NULL_BLOCK_ID=0, offset 0).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_reqs
    valid = offs < n
    req = tl.load(idx_mapping_ptr + offs, mask=valid, other=-1)
    rmask = valid & (req >= 0)
    req_safe = tl.maximum(req, 0)
    # Gather: padding/filtered rows become fresh (-1 col, 0 off).
    ssm_col = tl.where(
        rmask, tl.load(src_ssm_col_ptr + req_safe, mask=rmask, other=-1), -1
    )
    conv_col = tl.where(
        rmask, tl.load(conv_src_col_ptr + req_safe, mask=rmask, other=-1), -1
    )
    off = tl.where(rmask, tl.load(conv_src_off_ptr + req_safe, mask=rmask, other=0), 0)
    # Resolve: col < 0 -> NULL(0); else block_table[row, clamp(col)]. Block-table
    # loads use rmask (real cols, offs < n) not mask (offs < num_reqs): padded
    # phys is discarded anyway, and this lets num_reqs exceed the block-table row
    # count (resolving into a per-token CUDAGraph buffer) without an OOB read.
    base = block_table_ptr + offs.to(tl.int64) * bt_stride0
    ssm_c = tl.minimum(tl.maximum(ssm_col, 0), max_col).to(tl.int64)
    conv_c = tl.minimum(tl.maximum(conv_col, 0), max_col).to(tl.int64)
    ssm_phys = tl.load(base + ssm_c * bt_stride1, mask=rmask, other=0)
    conv_phys = tl.load(base + conv_c * bt_stride1, mask=rmask, other=0)
    tl.store(ssm_phys_ptr + offs, tl.where(ssm_col >= 0, ssm_phys, 0), mask=mask)
    tl.store(conv_phys_ptr + offs, tl.where(conv_col >= 0, conv_phys, 0), mask=mask)
    tl.store(conv_off_ptr + offs, off, mask=mask)


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
    spec_ssm_src_state_indices: torch.Tensor | None = None  # shape: [num_spec_decodes,]
    non_spec_ssm_src_state_indices: torch.Tensor | None = None  # shape: [num_decodes,]

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
    # Chunk-kernel inputs for prefill
    prefill_query_start_loc: torch.Tensor | None = None
    prefill_state_indices: torch.Tensor | None = None
    prefill_has_initial_state: torch.Tensor | None = None

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

        # Copy-free align (V1 and V2): the forward kernel reads the previous
        # running state directly from its source block instead of pre-copying it.
        # ``build`` resolves the src columns to physical block ids per step.
        self.use_align_mode: bool = vllm_config.cache_config.mamba_cache_mode == "align"

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
        # Persistent src buffers for full CUDAGraph (align only): SRC_PRERESOLVED
        # is a constexpr baked at capture, so the src must flow through these
        # fixed-address buffers -- a distinct object from the dst tensor, so the
        # `src is not dst` selector bakes SRC_PRERESOLVED=True (NULL-filled at
        # capture; each replay copies the real src into the same buffer).
        self.spec_ssm_src_state_indices_buf: torch.Tensor | None = None
        self.non_spec_ssm_src_state_indices_buf: torch.Tensor | None = None
        self.spec_conv_src_state_indices_buf: torch.Tensor | None = None
        self.non_spec_conv_src_state_indices_buf: torch.Tensor | None = None
        self.spec_conv_src_offset_buf: torch.Tensor | None = None
        self.non_spec_conv_src_offset_buf: torch.Tensor | None = None
        if self.use_align_mode:
            self.spec_ssm_src_state_indices_buf = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.non_spec_ssm_src_state_indices_buf = torch.empty(
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

    @staticmethod
    def _split_align_cache_src_info(
        src_full: torch.Tensor | None,
        spec_mask: torch.Tensor,
        include_non_spec: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Split full-batch mamba align-cache src indices into spec and
        non-spec portions. When include_non_spec=False (pure spec-only
        batch), skip the non-spec computation and return None for it."""
        if src_full is None:
            return None, None
        spec = src_full[spec_mask].contiguous()
        non_spec = src_full[~spec_mask].contiguous() if include_non_spec else None
        return spec, non_spec

    def _resolve_align_src_into(
        self,
        idx_mapping: torch.Tensor,
        src_ssm_col: torch.Tensor,
        conv_src_col: torch.Tensor,
        conv_src_off: torch.Tensor,
        block_table: torch.Tensor,
        out_ssm: torch.Tensor,
        out_conv: torch.Tensor,
        out_off: torch.Tensor,
        n: int,
        out_len: int,
    ) -> None:
        """Resolve the per-req src columns into ``out_*[:out_len]``: ``[:n]`` are
        the resolved real reqs and ``[n:out_len]`` are NULL-filled by the kernel,
        so one launch both resolves and stages. ``out_*`` are the persistent
        CUDAGraph buffers on the decode path, or a fresh full-batch tensor on the
        eager path. The columns resolve against the FULL (non-windowed) block
        table, since the previous running block can lie outside the window on a
        block-boundary crossing."""
        _gather_resolve_align_src_kernel[(triton.cdiv(out_len, 256),)](
            idx_mapping,
            src_ssm_col,
            conv_src_col,
            conv_src_off,
            block_table,
            block_table.stride(0),
            block_table.stride(1),
            block_table.size(1) - 1,
            out_ssm,
            out_conv,
            out_off,
            n,
            out_len,
            BLOCK=256,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        align_src_ssm_col: torch.Tensor | None = None,
        align_conv_src_col: torch.Tensor | None = None,
        align_conv_src_off: torch.Tensor | None = None,
        align_idx_mapping: torch.Tensor | None = None,
        align_num_reqs: int = 0,
        for_capture: bool = False,
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
                spec_sequence_masks = async_tensor_h2d(
                    spec_sequence_masks_cpu, device=query_start_loc.device
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
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        if num_prefills > 0:
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            # In a mixed non-spec batch, decodes are peeled off to the recurrent
            # kernel (decode-first front slice), so build chunk metadata from the
            # rebased prefill-only cu_seqlens; otherwise use the full non-spec one.
            # _forward_core keys off the same condition, so they agree.
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = (
                    non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                )
                prefill_query_start_loc_cpu = (
                    non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            if self.gdn_prefill_backend == "cutedsl":
                from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                    prepare_metadata_cutedsl,
                )

                assert prefill_query_start_loc is not None
                assert prefill_query_start_loc_cpu is not None
                total_tokens = int(prefill_query_start_loc_cpu[-1].item())
                chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
                    prefill_query_start_loc,
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

                assert prefill_query_start_loc_cpu is not None
                chunk_indices = async_tensor_h2d(
                    prepare_chunk_indices(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                    device=gpu_device,
                )
                chunk_offsets = async_tensor_h2d(
                    prepare_chunk_offsets(prefill_query_start_loc_cpu, FLA_CHUNK_SIZE),
                    device=gpu_device,
                )

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
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        # Function code counted on either presency non-spec decode or spec decode,
        # but not both.
        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        # Prepare per-request tensors for cudagraph. m.num_actual_tokens is
        # token-padded for FULL graph replay, but the GDN state/query/accepted
        # metadata below is indexed by request.
        batch_size = m.num_reqs

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

        # --- Mamba "align" copy-free src resolution --------------------------
        # Resolve the per-req src columns to physical block ids in spec/non-spec
        # order. The cg_spec / cg_nonspec conditions mirror the state-index
        # CUDAGraph staging above, so the src lands in the same buffer the SSM /
        # conv kernels read at replay; the eager / mixed path resolves into a
        # fresh full-batch tensor and splits it.
        spec_ssm_src_state_indices: torch.Tensor | None = None
        non_spec_ssm_src_state_indices: torch.Tensor | None = None
        spec_conv_src_state_indices: torch.Tensor | None = None
        non_spec_conv_src_state_indices: torch.Tensor | None = None
        spec_conv_src_offset: torch.Tensor | None = None
        non_spec_conv_src_offset: torch.Tensor | None = None
        if self.use_align_mode:
            # No cached state -> all-fresh NULL src. True at CUDAGraph capture and
            # on the profiling / warmup dummy runs (which never run
            # preprocess_mamba, so the runner has no columns to feed); a real
            # forward always carries the columns. Mirrors V2's for_capture path.
            no_src_state = for_capture or align_src_ssm_col is None
            cg_spec = (
                self.use_full_cuda_graph
                and num_prefills == 0
                and num_decodes == 0
                and num_spec_decodes <= self.decode_cudagraph_max_bs
                and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
            )
            cg_nonspec = (
                self.use_full_cuda_graph
                and num_prefills == 0
                and num_spec_decodes == 0
                and num_decodes <= self.decode_cudagraph_max_bs
            )
            if cg_spec:
                assert self.spec_ssm_src_state_indices_buf is not None
                assert self.spec_conv_src_state_indices_buf is not None
                assert self.spec_conv_src_offset_buf is not None
                if no_src_state:
                    # No cached state (capture / profiling): NULL-fill buffers.
                    self.spec_ssm_src_state_indices_buf[:batch_size].fill_(
                        NULL_BLOCK_ID
                    )
                    self.spec_conv_src_state_indices_buf[:batch_size].fill_(
                        NULL_BLOCK_ID
                    )
                    self.spec_conv_src_offset_buf[:batch_size].fill_(0)
                else:
                    assert (
                        align_idx_mapping is not None
                        and align_src_ssm_col is not None
                        and align_conv_src_col is not None
                        and align_conv_src_off is not None
                    )
                    self._resolve_align_src_into(
                        align_idx_mapping,
                        align_src_ssm_col,
                        align_conv_src_col,
                        align_conv_src_off,
                        m.block_table_tensor,
                        self.spec_ssm_src_state_indices_buf,
                        self.spec_conv_src_state_indices_buf,
                        self.spec_conv_src_offset_buf,
                        num_spec_decodes,
                        batch_size,
                    )
                spec_ssm_src_state_indices = self.spec_ssm_src_state_indices_buf[
                    :batch_size
                ]
                spec_conv_src_state_indices = self.spec_conv_src_state_indices_buf[
                    :batch_size
                ]
                spec_conv_src_offset = self.spec_conv_src_offset_buf[:batch_size]
            elif cg_nonspec:
                assert self.non_spec_ssm_src_state_indices_buf is not None
                assert self.non_spec_conv_src_state_indices_buf is not None
                assert self.non_spec_conv_src_offset_buf is not None
                if no_src_state:
                    # No cached state (capture / profiling): NULL-fill buffers.
                    self.non_spec_ssm_src_state_indices_buf[:batch_size].fill_(
                        NULL_BLOCK_ID
                    )
                    self.non_spec_conv_src_state_indices_buf[:batch_size].fill_(
                        NULL_BLOCK_ID
                    )
                    self.non_spec_conv_src_offset_buf[:batch_size].fill_(0)
                else:
                    assert (
                        align_idx_mapping is not None
                        and align_src_ssm_col is not None
                        and align_conv_src_col is not None
                        and align_conv_src_off is not None
                    )
                    self._resolve_align_src_into(
                        align_idx_mapping,
                        align_src_ssm_col,
                        align_conv_src_col,
                        align_conv_src_off,
                        m.block_table_tensor,
                        self.non_spec_ssm_src_state_indices_buf,
                        self.non_spec_conv_src_state_indices_buf,
                        self.non_spec_conv_src_offset_buf,
                        num_decodes,
                        batch_size,
                    )
                non_spec_ssm_src_state_indices = (
                    self.non_spec_ssm_src_state_indices_buf[:batch_size]
                )
                non_spec_conv_src_state_indices = (
                    self.non_spec_conv_src_state_indices_buf[:batch_size]
                )
                non_spec_conv_src_offset = self.non_spec_conv_src_offset_buf[
                    :batch_size
                ]
            elif not for_capture:
                # Eager / mixed path: no CUDAGraph decode buffer applies (prefill
                # / mixed / over capture size / non-cudagraph run). Resolve into a
                # fresh full-batch tensor and split it. The elif chain makes this
                # mutually exclusive with the cg_* branches above.
                block_table = m.block_table_tensor
                num_reqs_full = block_table.size(0)
                align_ssm_src_full = torch.empty(
                    num_reqs_full, dtype=torch.int32, device=block_table.device
                )
                align_conv_src_full = torch.empty(
                    num_reqs_full, dtype=torch.int32, device=block_table.device
                )
                align_conv_off_full = torch.empty(
                    num_reqs_full, dtype=torch.int32, device=block_table.device
                )
                if align_src_ssm_col is None:
                    # No cached state (profiling / warmup): all-fresh NULL src.
                    align_ssm_src_full.fill_(NULL_BLOCK_ID)
                    align_conv_src_full.fill_(NULL_BLOCK_ID)
                    align_conv_off_full.fill_(0)
                else:
                    assert align_conv_src_col is not None
                    assert align_conv_src_off is not None
                    assert align_idx_mapping is not None
                    self._resolve_align_src_into(
                        align_idx_mapping,
                        align_src_ssm_col,
                        align_conv_src_col,
                        align_conv_src_off,
                        block_table,
                        align_ssm_src_full,
                        align_conv_src_full,
                        align_conv_off_full,
                        align_num_reqs,
                        num_reqs_full,
                    )
                if spec_sequence_masks is None:
                    # Pure non-spec batch: rows are already in non-spec order.
                    non_spec_ssm_src_state_indices = align_ssm_src_full
                    non_spec_conv_src_state_indices = align_conv_src_full
                    non_spec_conv_src_offset = align_conv_off_full
                elif num_prefills == 0 and num_decodes == 0:
                    # Pure spec batch: the spec rows are the contiguous front
                    # prefix (padded rows are always at the back), so a view
                    # suffices instead of a boolean-mask gather.
                    spec_ssm_src_state_indices = align_ssm_src_full[:num_spec_decodes]
                    spec_conv_src_state_indices = align_conv_src_full[:num_spec_decodes]
                    spec_conv_src_offset = align_conv_off_full[:num_spec_decodes]
                else:
                    # Mixed batch: split by the spec mask.
                    assert spec_sequence_masks_cpu is not None
                    spec_ssm_src_state_indices, non_spec_ssm_src_state_indices = (
                        self._split_align_cache_src_info(
                            align_ssm_src_full, spec_sequence_masks_cpu
                        )
                    )
                    spec_conv_src_state_indices, non_spec_conv_src_state_indices = (
                        self._split_align_cache_src_info(
                            align_conv_src_full, spec_sequence_masks_cpu
                        )
                    )
                    spec_conv_src_offset, non_spec_conv_src_offset = (
                        self._split_align_cache_src_info(
                            align_conv_off_full, spec_sequence_masks_cpu
                        )
                    )

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
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            spec_ssm_src_state_indices=spec_ssm_src_state_indices,
            non_spec_ssm_src_state_indices=non_spec_ssm_src_state_indices,
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

        return self.build(
            0,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            for_capture=True,
        )
