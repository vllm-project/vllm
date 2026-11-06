# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online lm_head quantization: weight-only methods selected by
``quantization_config.lm_head``, and online-quantized copies of an
already-loaded lm_head."""

from abc import abstractmethod
from itertools import chain
from typing import ClassVar, Literal
from weakref import WeakKeyDictionary

import torch
from torch.nn import Module, Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
    init_nvfp4_linear_kernel,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineLinearMethod,
    OnlineLinearBase,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
    ref_nvfp4_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
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

    scheme: ClassVar[str]
    """The ``quantization_config.lm_head`` name, for logs and errors."""
    weight_type: ClassVar[ScalarType]
    group_size: ClassVar[int]
    """Input channels per scale; -1 for one scale per output channel."""
    zero_points: ClassVar[bool]
    n_align: ClassVar[int] = 1
    """Required divisor of the vocab shard (output rows)."""
    k_align: ClassVar[int] = 1
    """Required divisor of the hidden size (input columns)."""

    @abstractmethod
    def quantize(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Quantize ``weight`` [N, K] into the `MPLinearKernel` layout.

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
            # `weight` is the input embeddings' parameter: keep it, and add
            # the quantized copy under its own name.
            w_q_name = "weight_packed"
            w_q = Parameter(packed, requires_grad=False)
            set_weight_attrs(w_q, {"input_dim": 1, "output_dim": 0, "packed_dim": 1})
            layer.register_parameter(w_q_name, w_q)
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
            w_q_name = "weight"
            replace_parameter(layer, w_q_name, packed)
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


def _quantize_nvfp4(
    weight: torch.Tensor, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4-quantize the rows of `weight` without the SM100+ quant kernel.

    Returns packed E2M1 codes (low nibble first), E4M3 scales per 16 values and
    the fp32 dequantization global scale, in the checkpoint layout.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    global_scale = (weight_amax(weight).float() / (FLOAT4_E2M1_MAX * fp8_max)).clamp(
        min=1e-12
    )
    inv_global_scale = (1.0 / global_scale).reshape(())
    # E2M1 magnitudes 0, 0.5, ..., 6 doubled to integers, mapped to their codes.
    code_of = torch.zeros(13, dtype=torch.uint8, device=weight.device)
    code_of[torch.tensor([0, 1, 2, 3, 4, 6, 8, 12])] = torch.arange(
        8, dtype=torch.uint8
    ).to(weight.device)
    n, k = weight.shape
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(n, k // 16, dtype=torch.float8_e4m3fn, device=weight.device)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        values, block_scales = ref_nvfp4_quant(weight[rows], inv_global_scale, 16)
        codes = code_of[(values.abs() * 2).long()] | ((values < 0).to(torch.uint8) << 3)
        packed[rows] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[rows] = block_scales.to(torch.float8_e4m3fn)
    return packed, scales, global_scale


class _LMHeadNvfp4Method(OnlineLinearBase):
    """Weight-only NVFP4 lm_head copy on the platform's W4A16 kernel;
    activations stay in the model dtype."""

    def process_weights_after_loading(self, layer: Module) -> None:
        packed, scales, global_scale = _quantize_nvfp4(layer.weight)
        replace_parameter(layer, "weight", packed)
        replace_parameter(layer, "weight_scale", scales)
        replace_parameter(layer, "weight_global_scale", global_scale)
        self.kernel = init_nvfp4_linear_kernel(use_a16=True)
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


_LM_HEAD_METHODS: dict[str, type[OnlineLinearBase]] = {
    "fp8": Fp8PtpcOnlineLinearMethod,
    "nvfp4": _LMHeadNvfp4Method,
}


def _lm_head_method(quantization: str) -> type[OnlineLinearBase]:
    if quantization == "int4_per_group_32_asym":
        # online.int4 imports this module.
        from vllm.model_executor.layers.quantization.online.int4 import (
            Int4PerGroup32AsymLMHeadMethod,
        )

        return Int4PerGroup32AsymLMHeadMethod
    return _LM_HEAD_METHODS[quantization]


def _quantize_copy(lm_head: ParallelLMHead, quantization: str) -> ParallelLMHead:
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
    # Online methods take their output dtype from the default dtype, which is
    # only the model dtype while the model itself is being built.
    dtype = lm_head.weight.dtype
    with set_default_torch_dtype(dtype):
        head = ParallelLMHead(
            lm_head.num_embeddings,
            lm_head.embedding_dim,
            params_dtype=dtype,
            org_num_embeddings=lm_head.org_vocab_size,
            padding_size=lm_head.padding_size,
            prefix=lm_head.prefix,
            disable_tp=lm_head.disable_tp,
            quant_method=_lm_head_method(quantization)(),
        )
    # Never loaded: a new param sharing the source storage, never a copy of it.
    head.weight = torch.nn.Parameter(lm_head.weight.data, requires_grad=False)
    head.quant_method.process_weights_after_loading(head)
    # The copy is never loaded. The online loader wrapper captures the layer,
    # so keeping it would leave each copy to the cycle collector.
    for param in head.parameters(recurse=False):
        if hasattr(param, "weight_loader"):
            del param.weight_loader
    return head


# Each quantized copy maps to the head it is derived from. Weak keys, and no
# module attribute, so the copy never owns the source as a submodule.
_SOURCES: WeakKeyDictionary[Module, tuple[ParallelLMHead, str]] = WeakKeyDictionary()


def quantized_lm_head_copy(
    lm_head: ParallelLMHead,
    quantization: Literal["fp8", "nvfp4", "int4_per_group_32_asym"],
) -> ParallelLMHead:
    """A ParallelLMHead holding this rank's shard of `lm_head`, quantized.

    "fp8" uses per-row weight and per-token activation scales on the platform's
    FP8 linear kernel; "nvfp4" is weight-only on the
    platform's W4A16 kernel; "int4_per_group_32_asym" is weight-only on the
    platform's mixed-precision linear kernel. `lm_head` is left
    unchanged and no bf16 copy of its weight is made. The copy is derived state:
    `refresh_quantized_lm_heads` re-derives it after `lm_head` changes.
    """
    head = _quantize_copy(lm_head, quantization)
    # Keep it out of layerwise reload, which would re-run
    # process_weights_after_loading on the already-quantized weight.
    get_layerwise_info(head).reset()
    _SOURCES[head] = (lm_head, quantization)
    logger.info(
        "Quantized a %s copy of the lm_head (%d x %d per rank).",
        quantization,
        head.output_size_per_partition,
        head.input_size_per_partition,
    )
    return head


@torch.no_grad()
def refresh_quantized_lm_heads(model: Module) -> None:
    """Re-derive the quantized lm_head copies in `model` from their source heads.

    Copies in place, so CUDA graphs keep pointing at valid tensors.
    """
    for module in model.modules():
        if module not in _SOURCES:
            continue
        fresh = _quantize_copy(*_SOURCES[module])
        for name, tensor in chain(
            fresh.named_parameters(recurse=False), fresh.named_buffers(recurse=False)
        ):
            getattr(module, name).copy_(tensor)


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
