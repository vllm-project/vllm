# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online lm_head quantization: weight-only methods for the model's own
lm_head, selected by a `quantization_config.targets` pattern naming it, and
private quantized copies of an already-loaded lm_head for speculative-decoding
drafters (`draft_lm_head_quantization`)."""

import math
from abc import abstractmethod
from collections.abc import Callable
from itertools import chain
from typing import ClassVar, Literal
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
    OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4WeightOnlyOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.scalar_type import ScalarType
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class OnlineLMHeadMethodBase(OnlineLinearBase):
    """Weight-only lm_head quantized at load time and run on the platform's
    mixed-precision linear kernel (`choose_mp_linear_kernel`).

    A subclass sets the weight format and implements `quantize`. An untied
    head loads its fp16/bf16 weight like any online linear layer and replaces
    it with the quantized copy. A tied head shares the input embeddings'
    weight, which stays unchanged: the quantized copy is added next to it.
    """

    supports_lm_head = True
    scheme: ClassVar[str]
    """The weight name in `QUANT_KEY_NAMES`, for logs and errors."""
    weight_type: ClassVar[ScalarType]
    group_size: ClassVar[int]
    """Input channels per scale; -1 for one scale per output channel."""
    zero_points: ClassVar[bool]
    n_align: ClassVar[int] = 1
    """Required divisor of the vocab shard (output rows)."""
    k_align: ClassVar[int] = 1
    """Required divisor of the hidden size (input columns)."""

    @classmethod
    def resolve_activation_quant_key(
        cls, activation_quant_key: QuantKey | None, explicit: bool
    ) -> QuantKey | None:
        # Weight-only: an explicit `activation: null` restates the default.
        if activation_quant_key is not None:
            raise NotImplementedError(
                f"online {cls.__name__} is weight-only, use `activation: null` "
                f"or omit it, got activation={activation_quant_key}"
            )
        return None

    @abstractmethod
    def quantize(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Quantize `weight` [N, K] into the `MPLinearKernel` layout.

        Returns the weight packed along K, the scales [N, K // group_size]
        ([N, 1] per channel) and the zero points packed along N, or None
        without zero points.
        """
        raise NotImplementedError

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        # Fail before loading the checkpoint, not after.
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"lm_head quantization {self.scheme!r} needs an fp16/bf16 "
                f"lm_head, got {params_dtype}."
            )
        if (
            input_size_per_partition % self.k_align
            or output_size_per_partition % self.n_align
        ):
            raise ValueError(
                f"lm_head quantization {self.scheme!r} needs a hidden size "
                f"divisible by {self.k_align} and a vocab shard divisible by "
                f"{self.n_align}, got {output_size_per_partition} x "
                f"{input_size_per_partition}."
            )
        self.mp_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.weight_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=self.zero_points,
        )
        # Raises if no kernel on this platform can run the layer.
        self.kernel_cls: type[MPLinearKernel] = choose_mp_linear_kernel(self.mp_config)
        self.kernel: MPLinearKernel | None = None
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def tie_weights(self, layer: Module, embed_tokens: Module) -> Module:
        # The head reads the embeddings' loaded weight. Nothing is loaded into
        # the head itself, so it leaves layerwise loading; the model loader's
        # process_weights_after_loading pass quantizes it.
        layer.weight = embed_tokens.weight
        get_layerwise_info(layer).reset()
        layer.online_lm_head_tied_weight = True
        return layer

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        tied = getattr(layer, "online_lm_head_tied_weight", False)
        weight = self.get_weight_for_quantization(layer)
        n, k = weight.shape
        packed, scales, zero_points = self.quantize(weight)
        set_weight_attrs(packed, {"packed_dim": 1})
        set_weight_attrs(scales, {"input_dim": 1, "output_dim": 0})
        if tied:
            # `weight` is the input embeddings' parameter, which they keep: the
            # head drops it, since kernels may re-register their packed weight
            # under that name.
            del layer.weight
            w_q = Parameter(packed, requires_grad=False)
            set_weight_attrs(w_q, {"input_dim": 1, "output_dim": 0, "packed_dim": 1})
            layer.register_parameter("weight", w_q)
            added = packed.nbytes + scales.nbytes
            if zero_points is not None:
                added += zero_points.nbytes
            logger.info_once(
                "The lm_head is tied to the input embeddings, which keep their "
                "%s weight: the %s lm_head adds %.2f GiB per rank instead of "
                "saving memory.",
                weight.dtype,
                self.scheme,
                added / 2**30,
            )
        else:
            # The loaded parameter already reports input_dim=1, output_dim=0.
            replace_parameter(layer, "weight", packed)
        w_q_name = "weight"
        replace_parameter(layer, "weight_scale", scales)
        w_zp_name = None
        if zero_points is not None:
            set_weight_attrs(
                zero_points, {"input_dim": 1, "output_dim": 0, "packed_dim": 0}
            )
            w_zp_name = "weight_zero_point"
            replace_parameter(layer, w_zp_name, zero_points)

        self.kernel = self.kernel_cls(
            self.mp_config,
            w_q_param_name=w_q_name,
            w_s_param_name="weight_scale",
            w_zp_param_name=w_zp_name,
        )
        self.kernel.process_weights_after_loading(layer)
        self.release_requantization_source_weights(layer)
        logger.info_once(
            "Quantized the lm_head to %s (%d x %d per rank) with %s.",
            self.scheme,
            n,
            k,
            self.kernel_cls.__name__,
        )
        layer._already_called_process_weights_after_loading = True

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert self.kernel is not None
        return self.kernel.apply_weights(layer, x, bias)


_LM_HEAD_METHODS: dict[str, type[OnlineLinearBase]] = {
    "fp8": Fp8PtpcOnlineLinearMethod,
    "nvfp4": Nvfp4WeightOnlyOnlineLinearMethod,
}


def gather_org_rows(lm_head: VocabParallelEmbedding) -> torch.Tensor:
    """The `org_vocab_size` rows of an unquantized `lm_head`, gathered from the
    TP ranks that shard it."""
    weight = lm_head.weight.data
    if lm_head.tp_size > 1:
        weight = tensor_model_parallel_all_gather(
            weight[: lm_head.shard_indices.num_org_elements_padded], dim=0
        )
    return weight[: lm_head.org_vocab_size]


def _quantize_copy(
    lm_head: VocabParallelEmbedding, quantization: str, num_rows: int | None = None
) -> ParallelLMHead:
    if (
        not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        )
        or getattr(lm_head, "bias", None) is not None
        or lm_head.weight.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "Quantizing an lm_head copy needs an unquantized fp16/bf16 lm_head "
            "without bias."
        )
    rows, org_rows = lm_head.num_embeddings, lm_head.org_vocab_size
    padding_size = lm_head.padding_size
    if num_rows is not None:
        rows = org_rows = num_rows
        # The first rows are split over the TP ranks anew, so they must divide.
        padding_size = math.lcm(padding_size, lm_head.tp_size)
    # Online methods take their output dtype from the default dtype, which is
    # only the model dtype while the model itself is being built.
    dtype = lm_head.weight.dtype
    with set_default_torch_dtype(dtype):
        head = ParallelLMHead(
            rows,
            lm_head.embedding_dim,
            params_dtype=dtype,
            org_num_embeddings=org_rows,
            padding_size=padding_size,
            prefix=lm_head.prefix,
            disable_tp=lm_head.disable_tp,
            quant_method=_LM_HEAD_METHODS[quantization](),
        )
    # Never loaded: a new param sharing the source storage (its first rows,
    # with num_rows), never a copy of it. With num_rows and TP, this rank's
    # shard of those rows is gathered from the ranks that hold them.
    if num_rows is None:
        weight = lm_head.weight.data
    else:
        start = head.shard_indices.padded_org_vocab_start_index
        weight = gather_org_rows(lm_head)[start:]
    weight = weight[: head.weight.shape[0]]
    if weight.shape[0] < head.weight.shape[0]:
        weight = F.pad(weight, (0, 0, 0, head.weight.shape[0] - weight.shape[0]))
    head.weight = torch.nn.Parameter(weight, requires_grad=False)
    head.quant_method.process_weights_after_loading(head)
    # The copy is never loaded. The online loader wrapper captures the layer,
    # so keeping it would leave each copy to the cycle collector.
    for param in head.parameters(recurse=False):
        if hasattr(param, "weight_loader"):
            del param.weight_loader
    return head


# Each quantized copy maps to the head and row count it is derived from. Weak
# keys, and no module attribute, so the copy never owns the source as a
# submodule.
_SOURCES: WeakKeyDictionary[Module, tuple[VocabParallelEmbedding, str, int | None]] = (
    WeakKeyDictionary()
)


def quantized_lm_head_copy(
    lm_head: VocabParallelEmbedding,
    quantization: Literal["fp8", "nvfp4"],
    num_rows: int | None = None,
) -> ParallelLMHead:
    """A ParallelLMHead holding this rank's shard of `lm_head`, quantized.

    "fp8" uses per-row weight and per-token activation scales on the platform's
    FP8 linear kernel; "nvfp4" is weight-only on the
    platform's W4A16 kernel. `lm_head` is left
    unchanged and no bf16 copy of its weight is made. The copy is derived state:
    `refresh_quantized_lm_heads` re-derives it after `lm_head` changes.

    With `num_rows`, the copy is a head over just the first `num_rows` rows of
    `lm_head`, padded and split over TP ranks like any lm_head. Building or
    refreshing it is then a collective over the TP group.
    """
    head = _quantize_copy(lm_head, quantization, num_rows)
    # Keep it out of layerwise reload, which would re-run
    # process_weights_after_loading on the already-quantized weight.
    get_layerwise_info(head).reset()
    _SOURCES[head] = (lm_head, quantization, num_rows)
    logger.info(
        "Quantized a %s copy of the lm_head (%d x %d per rank).",
        quantization,
        head.output_size_per_partition,
        head.input_size_per_partition,
    )
    return head


@torch.no_grad()
def refresh_quantized_lm_head(head: Module) -> None:
    """Re-derive one `quantized_lm_head_copy` from its source head, in place."""
    fresh = _quantize_copy(*_SOURCES[head])
    for name, tensor in chain(
        fresh.named_parameters(recurse=False), fresh.named_buffers(recurse=False)
    ):
        getattr(head, name).copy_(tensor)


# Other state derived from a source lm_head, keyed by the module that owns it:
# re-derived with the quantized copies, see register_lm_head_refresh.
_REFRESH_HOOKS: WeakKeyDictionary[Module, Callable[[], None]] = WeakKeyDictionary()


def register_lm_head_refresh(module: Module, refresh: Callable[[], None]) -> None:
    """Have `refresh_quantized_lm_heads` call `refresh` on any model holding
    `module`, after weight updates of the source lm_head."""
    _REFRESH_HOOKS[module] = refresh


@torch.no_grad()
def refresh_quantized_lm_heads(model: Module) -> None:
    """Re-derive the quantized lm_head copies in `model`, and the state
    registered with `register_lm_head_refresh`, from their source heads.

    Copies in place, so CUDA graphs keep pointing at valid tensors.
    """
    for module in model.modules():
        if module in _SOURCES:
            refresh_quantized_lm_head(module)
        if (refresh := _REFRESH_HOOKS.get(module)) is not None:
            refresh()


# (parent, attribute, copy) for each copy swapped out by share_source_lm_heads.
_SWAPPED: dict[Module, list[tuple[Module, str, Module]]] = {}


def share_source_lm_heads(model: Module) -> None:
    """Replace each quantized copy in `model` with its source head.

    `model` then loads weights like a drafter sharing the unquantized head,
    until `restore_quantized_lm_heads` swaps the copies back in.
    """
    swapped = [
        (module, name, child)
        for module in model.modules()
        for name, child in module.named_children()
        if child in _SOURCES
    ]
    for module, name, child in swapped:
        setattr(module, name, _SOURCES[child][0])
    _SWAPPED[model] = swapped


def restore_quantized_lm_heads(model: Module, refresh: bool = True) -> None:
    """Undo `share_source_lm_heads`, then re-derive the copies in `model`.

    Args:
        model: The model passed to `share_source_lm_heads`, if any.
        refresh: Whether to re-derive the copies; False when the source heads
            may be mid-reload.

    """
    for module, name, head in _SWAPPED.pop(model, ()):
        setattr(module, name, head)
    if refresh:
        refresh_quantized_lm_heads(model)
