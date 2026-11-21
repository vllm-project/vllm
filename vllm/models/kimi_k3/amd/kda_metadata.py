# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Kimi-K3 specialization of GDN attention metadata.

The request classification and cudagraph staging intentionally mirror
`GDNAttentionMetadataBuilder`. Only the FLA chunk metadata is built
differently on device rather than on the host.

With `--use-replayssm`, the spec rows also carry the FlyDSL verify's rows and
the commit that runs after sampling (`KDAReplaySSMVerifyMetadata`).
"""

from dataclasses import dataclass, fields, replace

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.models.kimi_k3.amd.ops.kda_replayssm import (
    KDAReplaySSMCommit,
    use_kda_replayssm,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import max_decode_query_len
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)

_BLOCK_T = 256
_MIN_BLOCK_N = 128

# block_m of the fused prefill conv1d in kda.py; shared GDN builder prebuilds only 8
PREFILL_CONV_BLOCK_M = 128


@triton.jit(do_not_specialize=["N"])
def _chunk_metadata_kernel(
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    N,
    BT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    i_n = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    is_seq = offs_n < N
    bos = tl.load(cu_seqlens + offs_n, mask=is_seq, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens + offs_n + 1, mask=is_seq, other=0).to(tl.int32)
    nt = tl.where(is_seq, tl.cdiv(eos - bos, BT), 0)

    base = tl.sum(tl.where(offs_n < i_n, nt, 0))
    num_chunks = tl.sum(tl.where(offs_n == i_n, nt, 0))

    tl.store(chunk_offsets + i_n, base)
    if i_n == 0:
        tl.store(chunk_offsets + N, tl.sum(nt))

    for t0 in range(0, num_chunks, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < num_chunks
        row = (base + offs_t) * 2
        tl.store(chunk_indices + row, tl.full([BLOCK_T], i_n, tl.int32), mask=mask_t)
        tl.store(chunk_indices + row + 1, offs_t.to(tl.int32), mask=mask_t)


def prepare_chunk_metadata_device(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FLA chunk metadata on device, with no host<->device transfer."""
    num_seqs = cu_seqlens_cpu.numel() - 1
    seq_lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    num_chunks = int(((seq_lens + chunk_size - 1) // chunk_size).sum())

    chunk_indices = torch.empty(
        num_chunks, 2, dtype=cu_seqlens.dtype, device=cu_seqlens.device
    )
    chunk_offsets = torch.empty(
        num_seqs + 1, dtype=torch.int64, device=cu_seqlens.device
    )
    _chunk_metadata_kernel[(num_seqs,)](
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        num_seqs,
        BT=chunk_size,
        BLOCK_N=max(_MIN_BLOCK_N, next_power_of_2(num_seqs + 1)),
        BLOCK_T=_BLOCK_T,
        num_warps=4,
    )
    return chunk_indices, chunk_offsets


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if PIN_MEMORY:
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


def build_conv1d_chunk_schedule(
    query_start_loc_cpu: torch.Tensor, block_m: int, device: torch.device
) -> dict[str, int | torch.Tensor]:
    """Host-built AITER conv1d chunk schedule: no D2H sync, own buffers."""
    assert query_start_loc_cpu.device.type == "cpu"
    nums = (-(-query_start_loc_cpu.diff() // block_m)).to(torch.int64)
    tot = int(nums.sum())
    batch_ptr = torch.repeat_interleave(
        torch.arange(nums.numel(), dtype=torch.int32), nums
    )
    starts = torch.repeat_interleave(nums.cumsum(0) - nums, nums)
    token_chunk_offset_ptr = (torch.arange(tot, dtype=torch.int64) - starts).to(
        torch.int32
    )
    return {
        "tot": tot,
        "batch_ptr": _to_device(batch_ptr, device),
        "token_chunk_offset_ptr": _to_device(token_chunk_offset_ptr, device),
    }


def _replayssm_dense_tokens(
    query_start_loc_cpu: torch.Tensor,
    spec_masks_cpu: torch.Tensor,
    rows: int,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-built maps between the batch's spec tokens and the verify's dense
    rows. A spec row shorter than `num_tokens` repeats its last token: the
    verify is causal, so the extra steps change none of the row's outputs, and
    the commit never folds past the accepted tokens. Padding rows read token 0.
    """
    starts = query_start_loc_cpu[:-1][spec_masks_cpu].to(torch.int64)
    lens = query_start_loc_cpu.diff()[spec_masks_cpu].to(torch.int64)
    steps = torch.arange(num_tokens, dtype=torch.int64)
    gather = starts[:, None] + torch.minimum(steps[None, :], (lens - 1)[:, None])
    gather = torch.nn.functional.pad(gather, (0, 0, 0, rows - lens.numel()))
    spec_rows = torch.repeat_interleave(torch.arange(lens.numel()), lens)
    offsets = torch.arange(int(lens.sum())) - torch.repeat_interleave(
        lens.cumsum(0) - lens, lens
    )
    return gather.flatten(), spec_rows * num_tokens + offsets


def _adaptive_verification_enabled(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    return speculative_config is not None and bool(
        getattr(speculative_config, "enable_adaptive_verification", False)
    )


@dataclass
class KDAReplaySSMVerifyMetadata:
    """The FlyDSL verify rows of one step and one KDA KV cache group.

    Spec row `b` owns the dense tokens `b * num_tokens` to
    `(b + 1) * num_tokens - 1`.
    """

    num_tokens: int
    # [rows] KDA page of each spec row, -1 for CUDA graph padding rows.
    cache_indices: torch.Tensor
    # [rows] intermediate_conv_window line of each spec row.
    line_indices: torch.Tensor
    # The first num_spec_decodes rows are live; the commit skips the rest.
    num_spec_decodes: int
    # [num_spec_decodes] batch row of each spec row, None when they coincide.
    request_rows: torch.Tensor | None
    commit: KDAReplaySSMCommit
    # When the spec tokens are not already dense at the front of the batch:
    # [rows * num_tokens] batch token of each dense token, and
    # [num_spec_decode_tokens] dense token of each spec token, in
    # `spec_token_indx` order.
    gather_tokens: torch.Tensor | None = None
    scatter_tokens: torch.Tensor | None = None


@dataclass
class KimiK3ROCmKDAMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
    replayssm: KDAReplaySSMVerifyMetadata | None = None

    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        r = self.replayssm
        if r is not None:
            r.commit(
                r.cache_indices,
                r.line_indices,
                r.request_rows,
                num_accepted_tokens,
                r.num_spec_decodes,
                r.num_tokens,
            )
        # Without prefix caching there is no running column to move.
        return None


class KimiK3ROCmKDAMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.use_replayssm = use_kda_replayssm(vllm_config)
        self._replayssm_commit: KDAReplaySSMCommit | None = None
        if self.use_replayssm:
            # The KDA page holds the checkpoint only: one block per request.
            self.spec_state_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs, 1), dtype=torch.int32, device=device
            )
            # One row, and one conv window line, per request.
            rows = vllm_config.scheduler_config.max_num_seqs
            # Static, so FULL graphs replay with this step's rows.
            self.replayssm_cache_indices = torch.full(
                (rows,), -1, dtype=torch.int32, device=device
            )
            self.replayssm_line_indices = torch.arange(
                rows, dtype=torch.int32, device=device
            )

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls: type["KimiK3ROCmKDAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> int | None:
        if not _adaptive_verification_enabled(vllm_config):
            return None
        return max_decode_query_len(vllm_config)

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        if self.use_replayssm and num_decode_draft_tokens_cpu is not None:
            # The verify takes at most num_spec + 1 tokens a row. Real spec rows
            # never have more; capture dummies can, and run as non-spec rows.
            too_wide = (
                common_attn_metadata.query_start_loc_cpu.diff() > self.num_spec + 1
            )
            if bool(too_wide.any()):
                num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu.masked_fill(
                    too_wide, -1
                )
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        if attn_metadata.nums_dict is not None:
            # Same non-spec query_start_loc the base builder schedules over.
            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            if attn_metadata.spec_sequence_masks_cpu is not None:
                query_lens_cpu = query_start_loc_cpu.diff()[
                    ~attn_metadata.spec_sequence_masks_cpu
                ]
                query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.numel() + 1, dtype=torch.int32
                )
                torch.cumsum(query_lens_cpu, dim=0, out=query_start_loc_cpu[1:])
            attn_metadata.nums_dict[PREFILL_CONV_BLOCK_M] = build_conv1d_chunk_schedule(
                query_start_loc_cpu,
                PREFILL_CONV_BLOCK_M,
                common_attn_metadata.query_start_loc.device,
            )
        if (
            attn_metadata.uniform_spec_sequence_length is not None
            and _adaptive_verification_enabled(self.vllm_config)
        ):
            attn_metadata = replace(attn_metadata, uniform_spec_sequence_length=None)
        if attn_metadata.checkpoint is not None:
            # The shared builder keys checkpoint rows on every request; the ROCm
            # prefill path indexes them by prefill_query_start_loc, so rebuild
            # with those rows.
            attn_metadata = replace(
                attn_metadata,
                checkpoint=self._build_checkpoint_metadata(
                    common_attn_metadata, attn_metadata, num_decode_draft_tokens_cpu
                ),
            )
        if self.use_replayssm and attn_metadata.spec_sequence_masks is not None:
            attn_metadata = self._build_replayssm_metadata(
                common_attn_metadata, attn_metadata
            )
        return attn_metadata

    def _get_replayssm_commit(self) -> KDAReplaySSMCommit:
        if self._replayssm_commit is None:
            context = self.vllm_config.compilation_config.static_forward_context
            self._replayssm_commit = KDAReplaySSMCommit(
                [context[name] for name in self.layer_names]
            )
        return self._replayssm_commit

    def _build_replayssm_metadata(
        self, m: CommonAttentionMetadata, attn_metadata: GDNAttentionMetadata
    ) -> KimiK3ROCmKDAMetadata:
        num_tokens = self.num_spec + 1
        num_spec_decodes = attn_metadata.num_spec_decodes
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        spec_masks_cpu = attn_metadata.spec_sequence_masks_cpu
        assert spec_state_indices is not None and spec_masks_cpu is not None
        spec_only = attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0
        # A spec-only batch keeps the shared builder's rows, FULL graph padding
        # included, and its spec rows lead the batch. A mixed batch has exactly
        # its spec rows, wherever they are.
        rows = spec_state_indices.shape[0] if spec_only else num_spec_decodes
        if rows > self.replayssm_line_indices.shape[0]:
            raise ValueError(
                f"{rows} KDA spec rows exceed max_num_seqs "
                f"({self.replayssm_line_indices.shape[0]})"
            )
        # The checkpoint is column 0. FlyDSL treats slot 0 as live, so the null
        # block of a padding row becomes -1.
        cache_indices = self.replayssm_cache_indices[:rows]
        live = cache_indices[:num_spec_decodes]
        live.copy_(spec_state_indices[:num_spec_decodes, 0])
        live.masked_fill_(live == NULL_BLOCK_ID, -1)
        cache_indices[num_spec_decodes:].fill_(-1)

        request_rows = None
        if not spec_only:
            request_rows = _to_device(
                spec_masks_cpu.nonzero().flatten().to(torch.int32),
                m.query_start_loc.device,
            )
        gather_tokens = scatter_tokens = None
        dense = (
            spec_only
            and attn_metadata.uniform_spec_sequence_length == num_tokens
            and m.num_actual_tokens >= rows * num_tokens
        )
        if not dense:
            gather_tokens, scatter_tokens = _replayssm_dense_tokens(
                m.query_start_loc_cpu, spec_masks_cpu, rows, num_tokens
            )
            gather_tokens = _to_device(gather_tokens, m.query_start_loc.device)
            scatter_tokens = _to_device(scatter_tokens, m.query_start_loc.device)

        return KimiK3ROCmKDAMetadata(
            **{f.name: getattr(attn_metadata, f.name) for f in fields(attn_metadata)},
            replayssm=KDAReplaySSMVerifyMetadata(
                num_tokens=num_tokens,
                cache_indices=cache_indices,
                line_indices=self.replayssm_line_indices[:rows],
                num_spec_decodes=num_spec_decodes,
                request_rows=request_rows,
                commit=self._get_replayssm_commit(),
                gather_tokens=gather_tokens,
                scatter_tokens=scatter_tokens,
            ),
        )

    def _build_checkpoint_metadata(
        self,
        m: CommonAttentionMetadata,
        attn_metadata: GDNAttentionMetadata,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
    ) -> MambaPrefillCheckpointMetadata | None:
        if attn_metadata.num_prefills == 0:
            return None
        # request_rows must line up with prefill_query_start_loc, either the
        # prefill tail of a decode-first batch, or every non-spec row
        # including padding
        num_decodes = attn_metadata.num_decodes
        request_rows = list(
            range(num_decodes, num_decodes + attn_metadata.num_prefills)
        )
        if attn_metadata.spec_sequence_masks is not None:
            assert num_decode_draft_tokens_cpu is not None
            request_rows = (
                (num_decode_draft_tokens_cpu < 0).nonzero().flatten().tolist()
            )
        return self.checkpoint_builder.build(m, request_rows)

    def _build_chunk_metadata(
        self,
        prefill_query_start_loc: torch.Tensor,
        prefill_query_start_loc_cpu: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_chunk_metadata_device(
            prefill_query_start_loc,
            prefill_query_start_loc_cpu,
            FLA_CHUNK_SIZE,
        )


class KimiK3ROCmKDABackend(GDNAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KIMI_K3_KDA_ROCM"

    @staticmethod
    def get_builder_cls() -> type[KimiK3ROCmKDAMetadataBuilder]:
        return KimiK3ROCmKDAMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True
