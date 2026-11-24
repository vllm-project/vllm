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
"""

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticChannelSym,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_gemv_bf16_fp4_gather
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.model_executor.kernels.linear import FP8ScaledMMLinearKernel

logger = init_logger(__name__)

_UNQUANTIZED = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
# On Qwen3.8 MTP, a rank-256 projection puts 97.5% of the greedy draft tokens
# in its top 16k rows.
_DYNAMIC_RANK = 256
_CHUNK = 16384
# Weights per E4M3 scale in NVFP4.
_NVFP4_GROUP = 16


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


def _first_rows(
    head: ParallelLMHead, weights: dict[str, torch.Tensor], num_rows: int
) -> ParallelLMHead:
    """The first `num_rows` rows of a quantized lm_head as their own lm_head,
    loaded from `weights`, the tensors `head` loaded before its kernel repacked
    them."""
    device = head.weight.device
    with set_default_torch_dtype(head.params_dtype), torch.device(device):
        rows = ParallelLMHead(
            num_rows,
            head.embedding_dim,
            params_dtype=head.params_dtype,
            quant_config=head.quant_config,
            prefix=head.prefix,
            disable_tp=True,
        )
    params = dict(rows.named_parameters())
    for name, weight in weights.items():
        if (param := params.get(name)) is None:
            # Skipped by the head's loader too, e.g. a weight-only input_scale.
            continue
        output_dim = getattr(param, "output_dim", None)
        if output_dim == 0 and getattr(param, "packed_dim", None) != 0:
            weight = weight[:num_rows]
        elif output_dim is not None:
            raise ValueError(
                f"draft_token_map cannot take rows of lm_head.{name} for "
                f"{type(rows.quant_method).__name__}."
            )
        param.weight_loader(param, weight.to(device))
    rows.quant_method.process_weights_after_loading(rows)
    return rows


def _modelopt_nvfp4(
    weights: dict[str, torch.Tensor], hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
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
        return None
    return packed, scales, global_scale.float().reshape(1)


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
        keep_quantized_head: Keep a quantized lm_head's loaded tensors in
            `keep_head_weights`, for a token map.

    """

    # Set by load_token_map.
    head: VocabParallelEmbedding
    static_head: ParallelLMHead
    # Set by load_token_map with dynamic rows.
    nvfp4: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    basis: torch.Tensor
    scorer: _Fp8Rows

    def __init__(
        self,
        logits_processor: LogitsProcessor,
        vocab_size: int,
        keep_quantized_head: bool = False,
    ):
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
        self.quantized = False
        self.keep_quantized_head = keep_quantized_head
        self.head_weights: dict[str, torch.Tensor] = {}
        # Packed rows, scales and global scale of a ModelOpt NVFP4 lm_head.
        self.checkpoint_nvfp4: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        self.checkpoint_nvfp4 = None

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
        return self.map_draft_to_target(top)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Target ids of draft-vocabulary ids from the pruned lm_head."""
        return draft_ids_to_target(draft_ids, self.draft_id_to_target_id)

    def keep_head_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        lm_head: nn.Module,
        prefix: str = "lm_head.",
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Pass `weights` through, keeping a quantized `lm_head`'s tensors on
        the CPU. Its kernel repacks them after loading, so `load_token_map`
        takes the rows from these instead."""
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
        self, lm_head: VocabParallelEmbedding, static_rows: int, dynamic_rows: int
    ) -> None:
        """Draft from the first `static_rows` rows of `lm_head`, plus
        `dynamic_rows` rows picked per draft token. The rows are an NVFP4 copy,
        or the checkpoint rows of a quantized `lm_head`. The target keeps
        verifying with the full `lm_head`."""
        head = getattr(lm_head, "base_layer", lm_head)
        if not isinstance(head, VocabParallelEmbedding):
            raise ValueError(f"draft_token_map does not support {type(head).__name__}.")
        lp = self.logits_processor
        loaded, self.head_weights = self.head_weights, {}
        quantized = not isinstance(head.quant_method, _UNQUANTIZED)
        dtype = head.params_dtype if quantized else head.weight.dtype
        if (
            self.draft_id_to_target_id is not None
            or getattr(head, "bias", None) is not None
            or dtype not in (torch.float16, torch.bfloat16)
            or lp.head_dtype not in (None, dtype)
            or lp.scale != 1.0
            or lp.soft_cap is not None
        ):
            raise ValueError(
                "draft_token_map needs an fp16/bf16 full-vocabulary lm_head "
                "without bias, logit scaling or soft capping, run in its dtype."
            )
        if quantized and not (loaded and isinstance(head, ParallelLMHead)):
            raise ValueError(
                "draft_token_map needs the loaded tensors of a quantized lm_head; "
                "the drafter must pass its weights through "
                "DraftVocab.keep_head_weights."
            )
        if head.tp_size > 1:
            raise ValueError("draft_token_map does not support tensor parallelism.")
        nvfp4 = _modelopt_nvfp4(loaded, head.embedding_dim) if quantized else None
        if dynamic_rows and not (
            dtype == torch.bfloat16
            and (nvfp4 is not None or not quantized)
            and current_platform.is_device_capability_family(120)
            and has_flashinfer_gemv_bf16_fp4_gather()
        ):
            raise ValueError(
                "draft_token_map_dynamic_rows needs a bf16 unquantized or "
                "ModelOpt NVFP4 lm_head, an SM12x GPU and FlashInfer's "
                "gemv_bf16_fp4_gather."
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
        self.quantized = quantized
        if quantized:
            self.static_head = _first_rows(head, loaded, static_rows)
            if nvfp4 is not None and dynamic_rows:
                packed, scales, global_scale = nvfp4
                device = head.weight.device
                self.checkpoint_nvfp4 = (
                    packed.to(device),
                    scales.to(device),
                    global_scale.to(device),
                )
        else:
            self.static_head = quantized_lm_head_copy(head, "nvfp4", static_rows)
        if dynamic_rows:
            self._build_dynamic()
        logger.info(
            "Draft vocabulary: first %d of %d tokens plus %d dynamic rows per "
            "draft token, %s.",
            static_rows,
            vocab_size,
            dynamic_rows,
            type(self.static_head.quant_method).__name__ if quantized else "NVFP4",
        )

    def _build_dynamic(self) -> None:
        """NVFP4 rows outside the static ones, and their FP8 scorer: the rows
        projected on the top right singular vectors of the lm_head (the top
        eigenvectors of `W^T W`), so `(x @ basis) @ scorer[i]` approximates the
        logit of row `i`."""
        from vllm.model_executor.layers.quantization.online.lm_head import (
            _quantize_nvfp4,
        )

        n, vocab_size = self.num_static_rows, self.head.org_vocab_size
        if self.checkpoint_nvfp4 is None:
            weight = self.head.weight[:vocab_size]
            packed, scales, global_scale = _quantize_nvfp4(weight[n:])
            self.nvfp4 = (packed, scales, global_scale.reshape(1))

            def rows_of(start: int, stop: int) -> torch.Tensor:
                return weight[start:stop]
        else:
            packed, scales, global_scale = self.checkpoint_nvfp4
            self.nvfp4 = (packed[n:], scales[n:], global_scale)

            def rows_of(start: int, stop: int) -> torch.Tensor:
                return dequantize_to_dtype(
                    packed[start:stop],
                    scales[start:stop],
                    global_scale,
                    torch.bfloat16,
                    _NVFP4_GROUP,
                    swizzle=False,
                )

        def blocks(start: int) -> Iterator[torch.Tensor]:
            for i in range(start, vocab_size, _CHUNK):
                yield rows_of(i, min(i + _CHUNK, vocab_size)).float()

        hidden_size = self.head.embedding_dim
        gram = torch.zeros(
            hidden_size, hidden_size, dtype=torch.float32, device=packed.device
        )
        for block in blocks(0):
            gram.addmm_(block.t(), block)
        basis = torch.linalg.eigh(gram).eigenvectors[:, -_DYNAMIC_RANK:].flip(-1)
        del gram
        self.basis = basis.to(torch.bfloat16).contiguous()
        scores = torch.cat([(block @ basis).bfloat16() for block in blocks(n)])
        self.scorer = _Fp8Rows(scores)

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
        out = static.new_full((static.shape[0], self.vocab_size), float("-inf"))
        out[:, :n] = static[:, :n]
        if not self.num_dynamic_rows:
            return out
        import flashinfer

        scores = self.scorer(hidden_states @ self.basis)
        picks = scores.topk(self.num_dynamic_rows, dim=-1, sorted=False).indices
        flashinfer.gemv_bf16_fp4_gather(
            hidden_states, *self.nvfp4, picks, out, out_col_offset=n
        )
        return out


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
    if not draft_vocab.quantized:
        # The rows of a quantized lm_head come from its checkpoint tensors.
        register_lm_head_refresh(model, draft_vocab.refresh)
