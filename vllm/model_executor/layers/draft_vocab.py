# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft lm_heads that cover a subset of the target vocabulary.

Checkpoints with a pruned draft lm_head (EAGLE3, DFlash, DSpark) ship
`draft_id_to_target_id` offsets. MTP drafters sharing the target's full lm_head
can instead be restricted with `draft_token_map_static_rows` (the first token
ids, which BPE tokenizers give to the most frequent tokens) and
`draft_token_map_dynamic_rows` (rows picked per draft token from a rank-256
projection of the lm_head). Either way `DraftVocab.compute_logits` returns
target-vocabulary logits that are `-inf` elsewhere, so speculators need no
special handling.

Under tensor parallelism the token map's rows are split evenly over the ranks:
each rank computes the logits of its share, and the ranks gather them.
"""

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
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
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_gemv_bf16_fp4_gather

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.model_executor.kernels.linear import FP8ScaledMMLinearKernel

logger = init_logger(__name__)

_UNQUANTIZED = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
# On Qwen3.8 MTP, a rank-256 projection puts 97.5% of the greedy draft tokens
# in its top 16k rows.
_DYNAMIC_RANK = 256
_CHUNK = 16384


class _Fp8Rows(ReplicatedLinear):
    """FP8 weight-only rows with one scale per row, for the scorer, which only
    ranks rows."""

    def __init__(self, weight: torch.Tensor):
        from vllm.model_executor.kernels.linear import init_wfp8_a16_linear_kernel

        num_rows = weight.shape[0]
        # Weight-only kernels can require aligned output rows.
        if padding := -num_rows % 8:
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
        # What a quantization scheme sets on its layer in create_weights.
        self.input_size_per_partition = self.input_size
        self.output_size_per_partition = self.output_size
        self.logical_widths = self.output_partition_sizes
        self.orig_dtype = self.params_dtype
        weight_fp8, weight_scale = ops.scaled_fp8_quant(
            weight, use_per_token_if_dynamic=True
        )
        self.kernel: FP8ScaledMMLinearKernel = init_wfp8_a16_linear_kernel(
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
        return self.kernel.apply_weights(self, x)[..., : self.num_rows]


def draft_logits_to_target(
    logits: torch.Tensor, draft_id_to_target_id: torch.Tensor | None, vocab_size: int
) -> torch.Tensor:
    """Scatter draft-vocabulary logits into target-vocabulary logits."""
    if draft_id_to_target_id is None:
        return logits
    base = torch.arange(logits.shape[-1], device=logits.device)
    targets = base + draft_id_to_target_id
    out = logits.new_full((logits.shape[0], vocab_size), float("-inf"))
    out[:, targets] = logits
    return out


def draft_ids_to_target(
    draft_ids: torch.Tensor, draft_id_to_target_id: torch.Tensor | None
) -> torch.Tensor:
    """Target ids of draft-vocabulary ids from a pruned lm_head."""
    if draft_id_to_target_id is None:
        return draft_ids
    return draft_ids + draft_id_to_target_id[draft_ids]


class DraftVocab:
    """Draft logits over a subset of the target vocabulary.

    Args:
        logits_processor: Logits processor of the draft lm_head.
        vocab_size: Width of the target-vocabulary logits. A smaller draft
            vocabulary gets a `draft_id_to_target_id` parameter for the model to
            register and load.

    """

    # Set by load_token_map.
    head: VocabParallelEmbedding
    static_head: ParallelLMHead
    # Set by load_token_map with dynamic rows.
    nvfp4: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    basis: torch.Tensor
    scorer: _Fp8Rows
    # Of the rows outside the static ones: how many each TP rank takes (the last
    # may get fewer), and the first of this rank's.
    rank_rows: int
    first_row: int

    def __init__(self, logits_processor: LogitsProcessor, vocab_size: int):
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size
        if logits_processor.vocab_size > vocab_size:
            raise ValueError(
                f"The draft vocabulary ({logits_processor.vocab_size}) is larger "
                f"than the target vocabulary ({vocab_size})."
            )
        self.draft_id_to_target_id: nn.Parameter | None = None
        if logits_processor.vocab_size < vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(logits_processor.vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        self.num_static_rows = 0
        self.num_dynamic_rows = 0

    def compute_logits(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Target-vocabulary logits, `-inf` outside the draft vocabulary."""
        if self.num_static_rows:
            return self._token_map_logits(hidden_states)
        logits = self.logits_processor(lm_head, hidden_states)
        return draft_logits_to_target(
            logits, self.draft_id_to_target_id, self.vocab_size
        )

    def get_top_tokens(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Greedy target token ids without materializing full logits."""
        if self.num_static_rows:
            return self._token_map_logits(hidden_states).argmax(dim=-1)
        top = self.logits_processor.get_top_tokens(lm_head, hidden_states)
        return draft_ids_to_target(top, self.draft_id_to_target_id)

    def load_token_map(
        self, lm_head: VocabParallelEmbedding, static_rows: int, dynamic_rows: int
    ) -> None:
        """Draft from an NVFP4 copy of the first `static_rows` rows of
        `lm_head`, plus `dynamic_rows` rows picked per draft token. The target
        keeps verifying with the full `lm_head`."""
        head = getattr(lm_head, "base_layer", lm_head)
        lp = self.logits_processor
        if (
            self.draft_id_to_target_id is not None
            or not isinstance(head, VocabParallelEmbedding)
            or not isinstance(head.quant_method, _UNQUANTIZED)
            or getattr(head, "bias", None) is not None
            or head.weight.dtype not in (torch.float16, torch.bfloat16)
            or lp.head_dtype not in (None, head.weight.dtype)
            or lp.scale != 1.0
            or lp.soft_cap is not None
        ):
            raise ValueError(
                "draft_token_map needs an unquantized fp16/bf16 full-vocabulary "
                "lm_head without bias, logit scaling or soft capping, run in its "
                "weight dtype."
            )
        if dynamic_rows and not (
            head.weight.dtype == torch.bfloat16
            and current_platform.is_device_capability_family(120)
            and has_flashinfer_gemv_bf16_fp4_gather()
        ):
            raise ValueError(
                "draft_token_map_dynamic_rows needs a bf16 lm_head, an SM12x GPU "
                "and FlashInfer's gemv_bf16_fp4_gather."
            )
        vocab_size = head.org_vocab_size
        static_rows = min(static_rows, vocab_size)
        if dynamic_rows > vocab_size - static_rows:
            raise ValueError(
                "draft_token_map_dynamic_rows must be at most "
                f"{vocab_size - static_rows}, got {dynamic_rows}."
            )
        from vllm.model_executor.layers.quantization.online.lm_head import (
            quantized_lm_head_copy,
        )

        self.head = head
        self.num_static_rows = static_rows
        self.num_dynamic_rows = dynamic_rows
        self.static_head = quantized_lm_head_copy(head, "nvfp4", static_rows)
        if dynamic_rows:
            self.rank_rows = math.ceil((vocab_size - static_rows) / head.tp_size)
            self.first_row = head.tp_rank * self.rank_rows
            self._build_dynamic()
        logger.info(
            "Draft vocabulary: first %d of %d tokens plus %d dynamic rows per "
            "draft token, NVFP4.",
            static_rows,
            vocab_size,
            dynamic_rows,
        )

    def _build_dynamic(self) -> None:
        """NVFP4 rows outside the static ones, and their FP8 scorer: the rows
        projected on the top right singular vectors of the lm_head (the top
        eigenvectors of `W^T W`), so `(x @ basis) @ scorer[i]` approximates the
        logit of row `i`.

        Each TP rank keeps its share of the rows. The global scale and the basis
        come from the whole lm_head, so they match tensor parallel size 1."""
        from vllm.model_executor.layers.quantization.online.lm_head import (
            gather_org_rows,
        )
        from vllm.model_executor.layers.quantization.online.nvfp4 import (
            _quantize_nvfp4,
        )

        weight = gather_org_rows(self.head)
        rows = weight[self.num_static_rows :]
        mine = slice(self.first_row, self.first_row + self.rank_rows)
        packed, scales, global_scale = _quantize_nvfp4(rows[mine], weight_amax(rows))
        self.nvfp4 = (packed, scales, global_scale.reshape(1))
        gram = weight.new_zeros(weight.shape[1], weight.shape[1], dtype=torch.float32)
        for block in weight.split(_CHUNK):
            block = block.float()
            gram.addmm_(block.t(), block)
        basis = torch.linalg.eigh(gram).eigenvectors[:, -_DYNAMIC_RANK:].flip(-1)
        del gram
        self.basis = basis.to(weight.dtype).contiguous()
        scores = torch.cat(
            [(block.float() @ basis).to(weight.dtype) for block in rows.split(_CHUNK)]
        )
        self.scorer = _Fp8Rows(scores[mine])

    def _dynamic_state(self) -> list[torch.Tensor]:
        return [*self.nvfp4, self.basis, *self.scorer.parameters()]

    @torch.no_grad()
    def refresh(self) -> None:
        """Re-derive the rows and the scorer in place after the lm_head
        changed, so CUDA graphs keep pointing at valid tensors."""
        from vllm.model_executor.layers.quantization.online.lm_head import (
            refresh_quantized_lm_head,
        )

        refresh_quantized_lm_head(self.static_head)
        if not self.num_dynamic_rows:
            return
        held, old = (self.nvfp4, self.basis, self.scorer), self._dynamic_state()
        self._build_dynamic()
        for dst, src in zip(old, self._dynamic_state(), strict=True):
            dst.copy_(src)
        self.nvfp4, self.basis, self.scorer = held

    def _token_map_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Target-vocabulary logits of the static and the picked rows."""
        n = self.num_static_rows
        static = self.static_head.quant_method.apply(self.static_head, hidden_states)
        if self.head.tp_size > 1:
            static = tensor_model_parallel_all_gather(static, dim=-1)
        out = static.new_full((static.shape[0], self.vocab_size), float("-inf"))
        out[:, :n] = static[:, :n]
        if self.num_dynamic_rows:
            self._add_dynamic_logits(hidden_states, out)
        return out

    def _add_dynamic_logits(
        self, hidden_states: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Write the logits of the rows picked per draft token into `out`."""
        import flashinfer

        n = self.num_static_rows
        tp_size = self.head.tp_size
        scores = self.scorer(hidden_states @ self.basis)
        if tp_size > 1:
            # Equal widths for the gather; the padding ends up past the vocab.
            scores = F.pad(scores, (0, self.rank_rows - scores.shape[-1]))
            scores = tensor_model_parallel_all_gather(scores, dim=-1)
            scores = scores[:, : self.head.org_vocab_size - n]
        picks = scores.topk(self.num_dynamic_rows, dim=-1, sorted=False).indices
        row_ids = picks
        if tp_size > 1:
            # Every rank picks the same rows and computes the ones it holds:
            # the kernel skips negative ids.
            row_ids = picks - self.first_row
            held = (row_ids >= 0) & (row_ids < self.nvfp4[0].shape[0])
            row_ids = torch.where(held, row_ids, -1)
        flashinfer.gemv_bf16_fp4_gather(
            hidden_states,
            *self.nvfp4,
            row_ids,
            out,
            out_col_offset=n + self.first_row,
        )
        if tp_size > 1:
            columns = n + picks
            # A rank without a picked row left -inf in its column.
            logits = tensor_model_parallel_all_gather(out.gather(1, columns), dim=-1)
            out.scatter_(1, columns, logits.unflatten(-1, (tp_size, -1)).amax(dim=1))


def load_draft_token_map(model: nn.Module, spec: "SpeculativeConfig") -> None:
    """Restrict a drafter's `DraftVocab` to the speculative config's token map."""
    from vllm.model_executor.layers.quantization.online.lm_head import (
        register_lm_head_refresh,
    )

    draft_vocab = getattr(model, "draft_vocab", None)
    if not isinstance(draft_vocab, DraftVocab):
        raise ValueError(f"draft_token_map is not supported by {type(model).__name__}.")
    assert spec.draft_token_map_static_rows is not None
    draft_vocab.load_token_map(
        model.lm_head,
        spec.draft_token_map_static_rows,
        spec.draft_token_map_dynamic_rows or 0,
    )
    register_lm_head_refresh(model, draft_vocab.refresh)
