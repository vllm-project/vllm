# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k kernels for the DSA sparse attention indexer."""

import functools

import torch

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import current_workspace_manager

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

try:
    from flashinfer.topk import (
        top_k_ragged_transform as _fi_top_k_ragged_transform,
    )
except ImportError:
    _fi_top_k_ragged_transform = None

# ---------------------------------------------------------------------------
# DeepSelect (vllm._deepselect_C)
# ---------------------------------------------------------------------------

# "auto" heuristic breakpoints, measured on GB200 and enforced by
# tests/kernels/test_top_k_per_row.py::test_sparse_indexer_topk_auto_is_fastest.

# Below this row count cooperative/persistent are faster than DeepSelect.
DEEP_SELECT_MIN_ROWS = 32

# cooperative_topk's hard row limit.
_COOPERATIVE_MAX_ROWS = 64

# topk at/below which DeepSelect wins from DEEP_SELECT_MIN_ROWS up; between
# this and _WIDE_TOPK cooperative wins below _COOPERATIVE_MAX_ROWS instead.
_COOPERATIVE_CROSSOVER_TOPK = 1024

# topk at which DeepSelect falls back to its maxtopk-4096 instantiation:
# cooperative wins within its row limit, FlashInfer wins in
# _FI_WIDE_TOPK_VOCAB_RANGE (past the row/row-count limits below), and
# persistent wins at shorter contexts.
_WIDE_TOPK = 2048
_FI_WIDE_TOPK_VOCAB_RANGE = (65536, 131072)
_FI_WIDE_TOPK_MAX_ROWS = 256

# Matches the -1 fill convention used for topk_indices_buffer elsewhere.
IDX_OOB_FILL_VALUE = -1

try:
    import vllm._deepselect_C  # noqa: F401  (registers torch.ops.deep_select)
except ImportError as e:
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "Failed to import the DeepSelect extension (vllm._deepselect_C): %s", e
    )


@functools.lru_cache(maxsize=1)
def get_deep_select_stride_requirement() -> tuple[int, int]:
    """Stride alignment requirement (input, output) in bytes."""
    return torch.ops.deep_select.get_alignment_requirement()


def is_deep_select_supported(input: torch.Tensor, topk: int) -> bool:
    """Whether the kernel accepts this input (dtype/stride/topk constraints)."""
    return (
        topk <= 4096
        and input.shape[1] < 2**23
        and input.dtype in (torch.float32, torch.bfloat16)
        and input.stride(1) == 1
        and input.stride(0)
        * input.element_size()
        % get_deep_select_stride_requirement()[0]
        == 0
    )


def _get_empty_and_aligned_tensor(
    dim0: int, dim1: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Tensor with shape (dim0, dim1) whose stride(0) is 32B-aligned."""
    output_stride_requirement = get_deep_select_stride_requirement()[1] // (
        dtype.itemsize
    )
    assert output_stride_requirement > 0
    dim1_rounded = (
        (dim1 + output_stride_requirement - 1) // output_stride_requirement
    ) * output_stride_requirement
    return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]


def deep_select_topk(
    input: torch.Tensor,
    topk: int,
    end: torch.Tensor | None = None,
    output_idx: torch.Tensor | None = None,
    indices_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Select the top-k indices per row of `input` with DeepSelect.

    Args:
        input: (num_rows, vocab_size), bf16 or fp32. stride(1) must be 1 and
            stride(0) must be 1024B-aligned.
        topk: Number of elements to select per row; must be <= 4096.
        end: Optional (num_rows,) int32 tensor with the exclusive right
            boundary of each row. Rows with `end[i] < topk` get their
            remaining indices filled with -1.
        output_idx: Optional preallocated (num_rows, topk) output tensor whose
            stride(0) is 32B-aligned (e.g. a slice of a wider buffer).
        indices_dtype: Output dtype when `output_idx` is not provided.

    Returns:
        The (num_rows, topk) indices tensor.
    """
    assert input.dim() == 2 and input.stride(1) == 1
    assert (
        input.stride(0) * input.element_size() % get_deep_select_stride_requirement()[0]
        == 0
    )

    num_rows = input.shape[0]
    if output_idx is None:
        output_idx = _get_empty_and_aligned_tensor(
            num_rows, topk, input.device, indices_dtype
        )
    else:
        assert output_idx.dtype == indices_dtype
        assert output_idx.shape[0] >= num_rows and output_idx.shape[1] >= topk

    torch.ops.deep_select.topk(
        input,
        topk,
        None,  # begin is not supported
        end,
        False,  # sorted_value
        False,  # sorted_index
        None,  # output_value
        output_idx,
        None,  # output_idx_offset
        IDX_OOB_FILL_VALUE,
        float("-inf"),  # value_oob_fill_value
        False,  # return_value
        True,  # abort_when_nan_found
    )
    return output_idx


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class SparseIndexerTopk(torch.nn.Module):
    """The sparse indexer's decode top-k stage.

    Selects among the available top-k kernels (see
    kernel_config.sparse_indexer_topk_backend) and runs the chosen one.
    """

    def __init__(self) -> None:
        super().__init__()
        kernel_config = get_current_vllm_config().kernel_config
        self._backend = kernel_config.sparse_indexer_topk_backend
        self._is_cuda = current_platform.is_cuda()
        self._has_deep_select = self._is_cuda and (
            current_platform.is_device_capability_family(100)
        )
        self._has_flashinfer_topk = _fi_top_k_ragged_transform is not None
        self._cooperative_capable = self._is_cuda and (
            current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )

    def resolve_backend(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> str:
        """Resolve the decode top-k implementation from the configured
        backend ("auto" heuristic chain, or a validated explicit value)."""
        if self._backend == "auto":
            return self._resolve_auto(logits, topk_tokens, num_rows)

        failures: list[str] = []
        if self._backend == "cooperative":
            failures = self._cooperative_constraints(logits, topk_tokens, num_rows)
        elif self._backend == "persistent":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            if topk_tokens not in (512, 1024, 2048):
                failures.append(
                    f"topk_tokens must be in (512, 1024, 2048), got {topk_tokens}"
                )
        elif self._backend == "deep_select":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            elif not self._has_deep_select:
                failures.append("requires SM100a/SM103a (10.x device family)")
            elif not is_deep_select_supported(logits, topk_tokens):
                failures.append(
                    f"inputs violate DeepSelect's constraints: dtype={logits.dtype},"
                    f" stride={logits.stride()}, topk={topk_tokens}"
                )
        elif self._backend == "flashinfer":
            if not self._is_cuda:
                failures.append("requires a CUDA platform")
            if not self._has_flashinfer_topk:
                failures.append(
                    "flashinfer.topk.top_k_ragged_transform is not importable"
                )
            if logits.dtype != torch.float32 or logits.stride(1) != 1:
                failures.append(
                    f"requires fp32 logits with stride(1) == 1, got "
                    f"dtype={logits.dtype}, stride={logits.stride()}"
                )
        if failures:
            raise RuntimeError(
                f"sparse_indexer_topk_backend='{self._backend}' was requested, but: "
                + "; ".join(failures)
            )
        return self._backend

    def _resolve_auto(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> str:
        """Pick the fastest applicable backend. Breakpoints are the named
        constants at module level (GB200 measurements)."""
        num_cols = logits.shape[1]
        coop_ok = not self._cooperative_constraints(logits, topk_tokens, num_rows)
        persistent_ok = self._is_cuda and topk_tokens in (512, 1024, 2048)

        if coop_ok and (
            topk_tokens >= _WIDE_TOPK
            or (
                topk_tokens >= _COOPERATIVE_CROSSOVER_TOPK
                and num_rows < _COOPERATIVE_MAX_ROWS
            )
            or num_rows < DEEP_SELECT_MIN_ROWS
        ):
            return "cooperative"
        if topk_tokens >= _WIDE_TOPK and num_rows > _COOPERATIVE_MAX_ROWS:
            lo, hi = _FI_WIDE_TOPK_VOCAB_RANGE
            if (
                lo < num_cols <= hi
                and num_rows <= _FI_WIDE_TOPK_MAX_ROWS
                and logits.is_contiguous()
                and self._has_flashinfer_topk
            ):
                return "flashinfer"
            if num_cols <= lo and persistent_ok:
                return "persistent"
        if (
            self._has_deep_select
            and is_deep_select_supported(logits, topk_tokens)
            and num_rows >= DEEP_SELECT_MIN_ROWS
        ):
            return "deep_select"
        if coop_ok:
            return "cooperative"
        if persistent_ok:
            return "persistent"
        return "per_row"

    def _cooperative_constraints(
        self, logits: torch.Tensor, topk_tokens: int, num_rows: int
    ) -> list[str]:
        """Unmet constraints of cooperative_topk (empty when applicable)."""
        failures = []
        if not self._is_cuda:
            failures.append("requires a CUDA platform")
        if topk_tokens not in (512, 1024, 2048):
            failures.append(
                f"topk_tokens must be in (512, 1024, 2048), got {topk_tokens}"
            )
        if num_rows > 64:
            failures.append(f"num_rows must be <= 64, got {num_rows}")
        if logits.stride(0) % 4 != 0:
            failures.append(
                f"logits.stride(0) must be divisible by 4, got {logits.stride(0)}"
            )
        if self._is_cuda and not self._cooperative_capable:
            failures.append("requires SM90+ and is not supported on the SM12x family")
        return failures

    @staticmethod
    def _row_ends(seq_lens: torch.Tensor, next_n: int, num_rows: int) -> torch.Tensor:
        """Per-row exclusive end offsets (int32, (num_rows,)) for top-k
        kernels that take ragged lengths (DeepSelect, FlashInfer, torch
        reference).

        seq_lens is (B, next_n) per-row effective lens for native spec decode
        and (B, 1) otherwise, in which case per-row lens are derived the same
        way as the other decode top-k kernels.
        """
        if seq_lens.numel() == num_rows:
            row_ends = seq_lens.reshape(-1)
        else:
            next_n_offsets = torch.arange(
                next_n, dtype=torch.int32, device=seq_lens.device
            )
            row_ends = (
                (seq_lens.reshape(-1, 1) - next_n + 1 + next_n_offsets)
                .clamp_(min=0)
                .reshape(-1)
            )
        assert row_ends.dtype == torch.int32
        return row_ends

    def forward(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        next_n: int,
        topk_indices: torch.Tensor,
        topk_tokens: int,
        max_seq_len: int,
    ) -> None:
        """Run the resolved decode top-k implementation, writing into
        topk_indices (int32, -1 fill for rows shorter than topk_tokens)."""
        backend = self.resolve_backend(logits, topk_tokens, logits.shape[0])
        if backend == "deep_select":
            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            deep_select_topk(logits, topk_tokens, end=row_ends, output_idx=topk_indices)
        elif backend == "cooperative":
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                max_seq_len,
            )
        elif backend == "persistent":
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        elif backend == "flashinfer":
            assert _fi_top_k_ragged_transform is not None
            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            offsets = torch.zeros(
                logits.shape[0], dtype=torch.int32, device=logits.device
            )
            # top_k_ragged_transform selects within [0, row_ends[i]) per
            # row and -1-fills past the row length.
            indices = _fi_top_k_ragged_transform(logits, offsets, row_ends, topk_tokens)
            topk_indices.copy_(indices)
        elif backend == "torch":
            # Debug reference: mask everything past each row's end, then topk.
            row_ends = self._row_ends(seq_lens, next_n, logits.shape[0])
            cols = torch.arange(logits.shape[1], device=logits.device)
            masked = logits.masked_fill(
                cols.unsqueeze(0) >= row_ends.unsqueeze(1), float("-inf")
            )
            indices = masked.topk(topk_tokens, dim=-1).indices
            in_range = torch.arange(topk_tokens, device=logits.device).unsqueeze(
                0
            ) < row_ends.unsqueeze(1)
            indices = torch.where(in_range, indices, -1)
            topk_indices.copy_(indices)
        else:
            assert backend == "per_row", f"unknown topk backend: {backend}"
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
