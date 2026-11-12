# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft lm_heads that cover a subset of the target vocabulary.

`DraftVocab` serves two kinds of drafters:

- Checkpoints with a pruned draft lm_head and `draft_id_to_target_id` offsets
  (EAGLE3, DFlash, DSpark). This used to live in each model file.
- MTP drafters that share the target's full lm_head, restricted at load time
  by `speculative_config.draft_token_map`. On bandwidth-bound GPUs the full
  head dominates the draft step, so each draft token computes exact logits
  only for:
  1. the listed rows, a fixed list of frequent tokens, and
  2. optionally `draft_token_map_dynamic_rows` more rows per draft token,
     picked by a rank-256 projection of the lm_head (built from its weights
     at load time), which recovers what the list misses, e.g. other languages.

Either way `compute_logits` returns target-vocabulary logits that are `-inf`
outside the subset, so speculators need no special handling. The target still
verifies with its full head, so only the acceptance length can change.
"""

import json
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
    _e2m1_inline,
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticChannelSym,
    weight_amax,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.kernels.linear import FP8ScaledMMLinearKernel

logger = init_logger(__name__)

_UNQUANTIZED = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
# Rank of the lm_head projection that scores the dynamic rows. On Qwen3.8
# MTP, rank 256 alone puts 97.5% of the greedy draft tokens in its top 16k.
_DYNAMIC_RANK = 256


def load_draft_token_ids(path: str, extra_ids: list[int]) -> torch.Tensor:
    """Load a draft token map and return its sorted, unique target ids.

    Args:
        path: SGLang `--speculative-token-map` file (`.pt`) or a JSON list.
        extra_ids: Ids that are always included, e.g. EOS.

    Returns:
        int64 CPU tensor of target token ids.

    """
    if path.endswith(".json"):
        with open(path) as f:
            ids = json.load(f)
    else:
        ids = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
    if not isinstance(ids, (list, tuple)):
        raise ValueError(f"draft_token_map {path!r} must hold a flat list of ids.")

    token_ids = torch.tensor([int(i) for i in ids] + extra_ids, dtype=torch.int64)
    if token_ids.numel() == 0:
        raise ValueError("draft_token_map is empty.")
    return torch.unique(token_ids, sorted=True)


# Rows per step when copying or quantizing lm_head rows, which bounds the
# temporaries.
_CHUNK = 16384


def _quantize_rows(
    weight: torch.Tensor, quantization: str, ids: torch.Tensor | None = None
) -> tuple[torch.Tensor, ...]:
    """Weight-only quantized `weight[ids]` (all rows by default).

    Returns `(fp8_rows, row_scales)` for "fp8" and
    `(packed_e2m1, e4m3_block_scales, global_scale)` for "nvfp4". The NVFP4
    global scale covers all of `weight`, so any subset quantizes the same way.
    """
    if ids is None:
        ids = torch.arange(weight.shape[0], device=weight.device)
    if quantization == "nvfp4":
        fp4_fp8_max = FLOAT4_E2M1_MAX * torch.finfo(torch.float8_e4m3fn).max
        global_scale = fp4_fp8_max / weight_amax(weight).float().clamp(min=1e-12)
    out: list[torch.Tensor] = []
    for start in range(0, ids.numel(), _CHUNK):
        rows = weight[ids[start : start + _CHUNK]]
        if quantization == "fp8":
            parts = ops.scaled_fp8_quant(rows, use_per_token_if_dynamic=True)
        else:
            parts = ops.scaled_fp4_quant(
                rows, global_scale, is_sf_swizzled_layout=False
            )
        if not out:
            out = [p.new_empty(ids.numel(), *p.shape[1:]) for p in parts]
        for dst, part in zip(out, parts):
            dst[start : start + part.shape[0]] = part
    return tuple(out) if quantization == "fp8" else (*out, 1.0 / global_scale)


class _Rows(ReplicatedLinear):
    """lm_head rows as a linear layer, optionally FP8 weight-only with one scale
    per row on the platform's weight-only kernel."""

    weight: nn.Parameter

    def __init__(self, weight: torch.Tensor, quantization: str | None):
        num_rows = weight.shape[0]
        # Weight-only kernels can require aligned output rows, which the
        # dynamic candidates need not be.
        if quantization is not None and (padding := -num_rows % 8):
            weight = F.pad(weight, (0, 0, 0, padding))
        with torch.device(weight.device):
            super().__init__(
                weight.shape[1],
                weight.shape[0],
                bias=False,
                params_dtype=weight.dtype,
                return_bias=False,
                disable_tp=True,
            )
        self.num_rows = num_rows
        self.kernel: FP8ScaledMMLinearKernel | None = None
        if quantization is None:
            self.weight_loader(self.weight, weight)
            self.quant_method.process_weights_after_loading(self)
            return
        assert quantization == "fp8"
        from vllm.model_executor.kernels.linear import init_wfp8_a16_linear_kernel

        # What a quantization scheme sets on its layer in create_weights.
        self.input_size_per_partition = self.input_size
        self.output_size_per_partition = self.output_size
        self.logical_widths = self.output_partition_sizes
        self.orig_dtype = self.params_dtype
        weight_fp8, weight_scale = _quantize_rows(weight, quantization)
        self.kernel = init_wfp8_a16_linear_kernel(
            weight_quant_key=kFp8StaticChannelSym,
            activation_quant_key=kFp8DynamicTensorSym,
            weight_shape=tuple(weight.shape),
            input_dtype=weight.dtype,
            out_dtype=weight.dtype,
        )
        # The scaled-mm kernels take the weight as (K, N), tagged as such.
        self.weight = nn.Parameter(weight_fp8.t(), requires_grad=False)
        self.weight.input_dim = 0
        self.weight.output_dim = 1
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.input_scale = None
        self.weight_block_size = None
        self.kernel.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.kernel is None:
            return self.quant_method.apply(self, x)
        return self.kernel.apply_weights(self, x)[..., : self.num_rows]


def _listed_rows(
    head: ParallelLMHead, weights: dict[str, torch.Tensor], ids: torch.Tensor
) -> ParallelLMHead:
    """The `ids` rows of a quantized lm_head as their own lm_head.

    Parameters indexed by output row are sliced from `weights`, the tensors
    `head` loaded before repacking, and the others are copied. The rows then
    go through the head's quantization method like any other loaded layer.
    """
    device = head.weight.device
    with set_default_torch_dtype(head.params_dtype), torch.device(device):
        rows = ParallelLMHead(
            ids.numel(),
            head.embedding_dim,
            params_dtype=head.params_dtype,
            quant_config=head.quant_config,
            prefix=head.prefix,
            disable_tp=True,
        )
    params = dict(rows.named_parameters())
    for name, weight in weights.items():
        param = params.get(name)
        if param is None:
            # Not a parameter of the head either, so its loader skipped it too
            # (e.g. the input_scale of a weight-only head).
            continue
        output_dim = getattr(param, "output_dim", None)
        if output_dim == 0 and getattr(param, "packed_dim", None) != 0:
            if weight.shape[0] != head.org_vocab_size:
                raise ValueError(
                    f"lm_head.{name} of shape {tuple(weight.shape)} is not "
                    "indexed by output row."
                )
            weight = weight.index_select(0, ids.to(weight.device))
        elif output_dim is not None:
            raise ValueError(
                f"draft_token_map cannot take rows of lm_head.{name} for "
                f"{type(rows.quant_method).__name__}."
            )
        param.weight_loader(param, weight.to(device))
    rows.quant_method.process_weights_after_loading(rows)
    return rows


# Weights per E4M3 scale in NVFP4.
_NVFP4_GROUP = 16


def _nvfp4_head(
    weights: dict[str, torch.Tensor], hidden_size: int
) -> tuple[torch.Tensor, ...]:
    """`(packed, block_scales, global_scale)` of a ModelOpt NVFP4 lm_head."""
    packed = weights.get("weight")
    scales = weights.get("weight_scale")
    global_scale = weights.get("weight_scale_2")
    if (
        packed is None
        or scales is None
        or global_scale is None
        or packed.dtype != torch.uint8
        or packed.shape[1] * 2 != hidden_size
        or scales.dtype != torch.float8_e4m3fn
        or scales.shape[1] * _NVFP4_GROUP != hidden_size
    ):
        raise ValueError(
            "draft_token_map_dynamic_rows supports quantized lm_heads only in "
            "the ModelOpt NVFP4 format."
        )
    return packed, scales, global_scale.float().reshape(1)


# The two kernels below compute `x[n] @ weight[ids[n, j]]` for the dynamic
# rows, reading only the picked rows instead of gathering them into a copy.
# One program handles BLOCK_ROWS picked rows of one token.
@triton.jit
def _gather_rows_gemv_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    ids_ptr,
    out_ptr,
    num_rows,
    hidden_size,
    weight_stride,
    HAS_SCALE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    token = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < num_rows
    ids = tl.load(ids_ptr + token * num_rows + rows, mask=row_mask, other=0)
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for start in range(0, hidden_size, BLOCK_HIDDEN):
        cols = start + tl.arange(0, BLOCK_HIDDEN)
        col_mask = cols < hidden_size
        x = tl.load(x_ptr + token * hidden_size + cols, mask=col_mask, other=0.0)
        w = tl.load(
            weight_ptr + ids[:, None].to(tl.int64) * weight_stride + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    if HAS_SCALE:
        acc *= tl.load(scale_ptr + ids, mask=row_mask, other=0.0)
    tl.store(
        out_ptr + token * num_rows + rows,
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask,
    )


@triton.jit
def _gather_rows_gemv_nvfp4_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    global_scale_ptr,
    ids_ptr,
    out_ptr,
    num_rows,
    hidden_size,
    packed_stride,
    scale_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    token = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < num_rows
    ids = tl.load(ids_ptr + token * num_rows + rows, mask=row_mask, other=0)
    ids = ids.to(tl.int64)
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    num_bytes = hidden_size // 2
    for start in range(0, num_bytes, BLOCK_BYTES):
        bcols = start + tl.arange(0, BLOCK_BYTES)
        bmask = bcols < num_bytes
        mask = row_mask[:, None] & bmask[None, :]
        # Each byte packs two E2M1 values (low nibble first) and every 8 bytes
        # (16 values) share one E4M3 block scale.
        b = tl.load(
            packed_ptr + ids[:, None] * packed_stride + bcols[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)
        sc = tl.load(
            scale_ptr + ids[:, None] * scale_stride + (bcols // 8)[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x_ptrs = x_ptr + token * hidden_size + 2 * bcols
        x_lo = tl.load(x_ptrs, mask=bmask, other=0.0).to(tl.float32)
        x_hi = tl.load(x_ptrs + 1, mask=bmask, other=0.0).to(tl.float32)
        v_lo = _e2m1_inline(b & 15)
        v_hi = _e2m1_inline((b >> 4) & 15)
        acc += tl.sum((v_lo * x_lo[None, :] + v_hi * x_hi[None, :]) * sc, axis=1)
    acc *= tl.load(global_scale_ptr)
    tl.store(
        out_ptr + token * num_rows + rows,
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask,
    )


def gather_rows_gemv(
    x: torch.Tensor, rows: tuple[torch.Tensor, ...], ids: torch.Tensor
) -> torch.Tensor:
    """`out[n, j] = x[n] @ weight[ids[n, j]]` without materializing the rows.

    Args:
        x: Activations, shape `(N, H)`.
        rows: `(weight,)` in the activation dtype, `(fp8_weight, row_scale)`
            or `(nvfp4_packed, block_scales, global_scale)`.
        ids: Row ids per token, shape `(N, J)`.

    """
    if not x.is_cuda:
        assert len(rows) == 1, "quantized rows need CUDA"
        return torch.einsum("nd,njd->nj", x, rows[0][ids])
    num_tokens, num_rows = ids.shape
    out = x.new_empty(num_tokens, num_rows)
    grid = (num_tokens, triton.cdiv(num_rows, 64))
    if len(rows) == 3:
        packed, scales, global_scale = rows
        _gather_rows_gemv_nvfp4_kernel[grid](
            x.contiguous(),
            packed,
            scales,
            global_scale,
            ids.contiguous(),
            out,
            num_rows,
            x.shape[-1],
            packed.stride(0),
            scales.stride(0),
            BLOCK_ROWS=64,
            BLOCK_BYTES=128,
        )
        return out
    weight = rows[0]
    _gather_rows_gemv_kernel[grid](
        x.contiguous(),
        weight,
        rows[1] if len(rows) == 2 else weight,
        ids.contiguous(),
        out,
        num_rows,
        x.shape[-1],
        weight.stride(0),
        HAS_SCALE=len(rows) == 2,
        BLOCK_ROWS=64,
        BLOCK_HIDDEN=128 if weight.element_size() > 1 else 256,
    )
    return out


def scatter_draft_logits(
    vocab_size: int,
    listed: torch.Tensor,
    token_ids: torch.Tensor,
    picked: torch.Tensor,
    picks: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Target-vocabulary logits, `-inf` outside the listed and picked rows.

    `out[n, token_ids] = listed[n]` and `out[n, candidate_ids[picks[n]]] =
    picked[n]`.
    """
    out = listed.new_full((listed.shape[0], vocab_size), float("-inf"))
    out[:, token_ids] = listed
    out.scatter_(1, candidate_ids[picks], picked)
    return out


class DraftVocab:
    """Draft logits over a subset of the target vocabulary.

    Args:
        logits_processor: Logits processor of the draft lm_head.
        vocab_size: Target vocabulary size. A smaller draft vocabulary gets a
            `draft_id_to_target_id` parameter for the model to register and
            load.
        keep_quantized_head: Keep a quantized lm_head's loaded tensors in
            `keep_head_weights`, for `speculative_config.draft_token_map`.

    """

    def __init__(
        self,
        logits_processor: LogitsProcessor,
        vocab_size: int,
        keep_quantized_head: bool = False,
    ):
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size
        self.draft_id_to_target_id: nn.Parameter | None = None
        if logits_processor.vocab_size < vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(logits_processor.vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        self.keep_quantized_head = keep_quantized_head
        # Loaded tensors of a quantized lm_head, see keep_head_weights.
        self.head_weights: dict[str, torch.Tensor] = {}
        # Set by load_token_map.
        self.token_ids: torch.Tensor | None = None
        self.rows: _Rows | ParallelLMHead | None = None
        self.num_dynamic_rows = 0

    def compute_logits(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Target-vocabulary logits, `-inf` outside the draft vocabulary."""
        if self.token_ids is not None:
            listed, picked = self._token_map_logits(hidden_states)
            if picked is not None:
                return scatter_draft_logits(
                    self.vocab_size, listed, self.token_ids, *picked, self.candidate_ids
                )
            out = listed.new_full((listed.shape[0], self.vocab_size), float("-inf"))
            out[:, self.token_ids] = listed
            return out

        logits = self.logits_processor(lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits
        base = torch.arange(logits.shape[-1], device=logits.device)
        targets = base + self.draft_id_to_target_id
        out = logits.new_full((logits.shape[0], self.vocab_size), float("-inf"))
        out[:, targets] = logits
        return out

    def get_top_tokens(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Greedy target token ids without materializing full logits."""
        if self.token_ids is not None:
            listed, picked = self._token_map_logits(hidden_states)
            listed_max, listed_col = listed.max(dim=-1)
            top = self.token_ids[listed_col]
            if picked is None:
                return top
            logits, picks = picked
            picked_max, picked_col = logits.max(dim=-1)
            pick = picks.gather(1, picked_col.unsqueeze(1)).squeeze(1)
            # Ties go to the listed row, as an argmax over [listed, picked] would.
            return torch.where(listed_max >= picked_max, top, self.candidate_ids[pick])

        top = self.logits_processor.get_top_tokens(lm_head, hidden_states)
        return self.map_draft_to_target(top)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Target ids of draft-vocabulary ids from the pruned lm_head."""
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def keep_head_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        lm_head: nn.Module,
        prefix: str = "lm_head.",
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Pass `weights` through, keeping a quantized `lm_head`'s tensors.

        Quantized lm_head kernels repack their weights after loading, so
        `load_token_map` takes the rows from these tensors instead.
        """
        keep = (
            self.keep_quantized_head
            and isinstance(lm_head, ParallelLMHead)
            and not isinstance(lm_head.quant_method, _UNQUANTIZED)
        )
        for name, weight in weights:
            if keep and name.startswith(prefix):
                self.head_weights[name.removeprefix(prefix)] = weight.to(
                    "cpu", copy=True
                )
            yield name, weight

    def load_token_map(
        self,
        lm_head: VocabParallelEmbedding,
        token_ids: torch.Tensor,
        dynamic_rows: int = 0,
        quantization: Literal["fp8", "nvfp4"] | None = None,
    ) -> None:
        """Restrict drafting to the `token_ids` rows of `lm_head`.

        The rows are copied out of `lm_head`, so the drafter keeps sharing it
        with the target, which verifies with the full vocabulary. A quantized
        `lm_head` gets its rows from the tensors kept by `keep_head_weights`, in
        their own quantized format. On a W4A4 head the dynamic rows run with
        BF16 activations, so only the acceptance length can change.

        Args:
            lm_head: The drafter's full-vocabulary lm_head.
            token_ids: Sorted, unique target ids of the listed rows.
            dynamic_rows: Rows outside the list picked per draft token.
            quantization: `draft_lm_head_quantization`, applied to the copied
                rows of an unquantized `lm_head` only. The listed rows are a
                quantized lm_head copy, see `quantized_lm_head_copy`; the
                dynamic rows are weight-only.

        """
        head = getattr(lm_head, "base_layer", lm_head)
        lp = self.logits_processor
        loaded, self.head_weights = self.head_weights, {}
        quantized = isinstance(head, VocabParallelEmbedding) and not isinstance(
            head.quant_method, _UNQUANTIZED
        )
        if (
            self.draft_id_to_target_id is not None
            or not isinstance(head, VocabParallelEmbedding)
            or getattr(head, "bias", None) is not None
            or lp.head_dtype
            not in (None, head.params_dtype if quantized else head.weight.dtype)
            or lp.scale != 1.0
            or lp.soft_cap is not None
        ):
            raise ValueError(
                "draft_token_map needs a full-vocabulary lm_head without bias, "
                "logit scaling or soft capping, run in its weight dtype."
            )
        if quantized and not (loaded and isinstance(head, ParallelLMHead)):
            raise ValueError(
                f"draft_token_map needs the loaded tensors of the quantized "
                f"lm_head, which {type(head.quant_method).__name__} repacks; the "
                "drafter must pass its weights through "
                "DraftVocab.keep_head_weights."
            )
        if quantized and quantization is not None:
            raise ValueError(
                "draft_lm_head_quantization applies to unquantized lm_heads. "
                "This lm_head is quantized in the checkpoint and its rows keep "
                "that format."
            )
        if head.tp_size > 1:
            raise ValueError("draft_token_map does not support tensor parallelism.")

        vocab_size = head.org_vocab_size
        bad = token_ids[(token_ids < 0) | (token_ids >= vocab_size)]
        if bad.numel() > 0:
            raise ValueError(
                f"draft_token_map has {bad.numel()} ids outside [0, {vocab_size}), "
                f"e.g. {bad[:5].tolist()}."
            )

        device = head.weight.device
        self.token_ids = token_ids.to(device)
        self.num_dynamic_rows = dynamic_rows
        if quantized:
            assert isinstance(head, ParallelLMHead)
            assert head.quant_config is not None
            self.rows = _listed_rows(head, loaded, token_ids)
            row_format: str = head.quant_config.get_name()
            if dynamic_rows > 0:
                self._init_quantized_dynamic_rows(head, loaded)
        else:
            weight = head.weight[:vocab_size]
            if quantization is None:
                self.rows = _Rows(weight[self.token_ids], None)
            else:
                from vllm.model_executor.layers.quantization.online.lm_head import (
                    quantized_lm_head_copy,
                )

                self.rows = quantized_lm_head_copy(head, quantization, self.token_ids)
            row_format = quantization or str(weight.dtype)
            if dynamic_rows > 0:
                self._init_dynamic_rows(
                    lambda ids: weight[ids], vocab_size, quantization is not None
                )
                # Quantized copies hold only the candidates and are indexed by
                # position.
                self.gather_rows = (
                    (weight,)
                    if quantization is None
                    else _quantize_rows(weight, quantization, self.candidate_ids)
                )
        logger.info(
            "Draft vocabulary: %d of %d tokens plus %d dynamic rows per draft "
            "token, %s rows.",
            token_ids.numel(),
            vocab_size,
            dynamic_rows,
            row_format,
        )

    def _init_quantized_dynamic_rows(
        self, head: ParallelLMHead, loaded: dict[str, torch.Tensor]
    ) -> None:
        """Dynamic rows of a ModelOpt NVFP4 lm_head, from its packed rows."""
        packed, scales, global_scale = _nvfp4_head(loaded, head.embedding_dim)
        device = head.weight.device
        dtype = head.params_dtype

        def rows_of(ids: torch.Tensor) -> torch.Tensor:
            ids = ids.cpu()
            return dequantize_to_dtype(
                packed[ids].to(device),
                scales[ids].to(device),
                global_scale.to(device),
                dtype,
                _NVFP4_GROUP,
                swizzle=False,
            )

        self._init_dynamic_rows(rows_of, head.org_vocab_size, True)
        ids = self.candidate_ids.cpu()
        self.gather_rows = (
            packed[ids].to(device),
            scales[ids].to(device),
            global_scale.to(device),
        )

    def _init_dynamic_rows(
        self,
        rows_of: Callable[[torch.Tensor], torch.Tensor],
        vocab_size: int,
        quantize_scorer: bool,
    ) -> None:
        """Score the rows outside the list with a rank-256 lm_head factor.

        The basis is the top right singular vectors of the lm_head (the top
        eigenvectors of `W^T W`), so
        `(x @ basis) @ (weight[i] @ basis)` approximates the logit of row `i`.
        The best `num_dynamic_rows` per token then get exact logits. The rows
        come from `rows_of(ids)` one chunk at a time.
        """
        assert self.token_ids is not None
        device = self.token_ids.device
        is_listed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        is_listed[self.token_ids] = True
        self.candidate_ids = (~is_listed).nonzero()[:, 0]
        if self.num_dynamic_rows > self.candidate_ids.numel():
            raise ValueError(
                "draft_token_map_dynamic_rows must be at most "
                f"{self.candidate_ids.numel()}, got {self.num_dynamic_rows}."
            )

        gram: torch.Tensor | None = None
        for ids in torch.arange(vocab_size, device=device).split(_CHUNK):
            rows = rows_of(ids)
            dtype = rows.dtype
            block = rows.float()
            if gram is None:
                gram = block.new_zeros(block.shape[1], block.shape[1])
            gram.addmm_(block.t(), block)
        assert gram is not None
        basis = torch.linalg.eigh(gram).eigenvectors[:, -_DYNAMIC_RANK:].flip(-1)
        del gram
        self.basis = basis.to(dtype).contiguous()
        scores = torch.cat(
            [
                (rows_of(ids).float() @ basis).to(dtype)
                for ids in self.candidate_ids.split(_CHUNK)
            ]
        )
        # The scorer only ranks rows, which FP8 preserves far better than NVFP4.
        self.scorer = _Rows(scores, "fp8" if quantize_scorer else None)

    def _token_map_logits(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Logits of the listed rows, and of the picked rows with the picks as
        positions in `candidate_ids`."""
        assert self.rows is not None and self.token_ids is not None
        rows = self.rows
        logits = rows.quant_method.apply(rows, hidden_states)
        logits = logits[..., : self.token_ids.numel()]
        if self.num_dynamic_rows == 0:
            return logits, None
        scores = self.scorer(hidden_states @ self.basis)
        picks = scores.topk(self.num_dynamic_rows, dim=-1, sorted=False).indices
        gather_ids = self.candidate_ids[picks] if len(self.gather_rows) == 1 else picks
        picked = gather_rows_gemv(hidden_states, self.gather_rows, gather_ids)
        return logits, (picked, picks)


def load_draft_token_map(model: nn.Module, vllm_config: "VllmConfig") -> None:
    """Restrict a drafter's `DraftVocab` to `speculative_config.draft_token_map`.

    EOS ids of the target are always included.
    """
    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_token_map is not None
    draft_vocab = getattr(model, "draft_vocab", None)
    if not isinstance(draft_vocab, DraftVocab):
        raise ValueError(f"draft_token_map is not supported by {type(model).__name__}.")
    model_config = vllm_config.model_config
    eos_ids: list[int] = []
    for eos in (
        getattr(model_config.hf_text_config, "eos_token_id", None),
        model_config.try_get_generation_config().get("eos_token_id"),
    ):
        if eos is not None:
            eos_ids.extend(eos if isinstance(eos, (list, tuple)) else [eos])
    draft_vocab.load_token_map(
        model.lm_head,
        load_draft_token_ids(spec.draft_token_map, eos_ids),
        spec.draft_token_map_dynamic_rows,
        spec.draft_lm_head_quantization,
    )
    if isinstance(draft_vocab.rows, ParallelLMHead):
        # A submodule, so that refresh_quantized_lm_heads re-derives it from
        # the target lm_head after weight updates.
        model.draft_token_map_head = draft_vocab.rows
