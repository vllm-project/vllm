# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers modeling backend support for linear attention layers."""

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.config.utils import getattr_iter
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    Mamba2SSM,
    Mixer2RMSNormGated,
    mamba_v2_sharded_weight_loader,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    SupportsMambaPrefixCaching,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.config import VllmConfig

VLLM_SSD_ATTR = "ssd"
"""Attribute in which to store the vLLM `Mamba2SSM` that serves a Transformers
mixer, mirroring `VLLM_ATTN_ATTR` for attention."""

VLLM_SSD_SIZES_ATTR = "vllm_ssd_sizes"
"""Attribute in which `shard_mamba2_mixer` stores the global `Mamba2Sizes` of a
Transformers mixer, whose own size attributes it makes rank-local."""

LINEAR_ATTENTION_LAYER_TYPES = ("linear_attention", "hybrid")
"""`layer_types` entries whose layer contains a linear attention mixer."""


@dataclass(frozen=True)
class Mamba2Sizes:
    """The sizes of a Transformers Mamba2 mixer before tensor parallelism."""

    intermediate_size: int
    n_groups: int
    num_heads: int


def _set_weight_loader(param: nn.Parameter, weight_loader) -> None:
    """Override the weight loader of `param`, which may already have one."""
    if isinstance(param, BasevLLMParameter):
        # Quantized layers (e.g. FP8), whose weight loader is a settable property
        param.weight_loader = weight_loader
        return
    if hasattr(param, "weight_loader"):
        # `set_weight_attrs` refuses to overwrite an attribute
        delattr(param, "weight_loader")
    set_weight_attrs(param, {"weight_loader": weight_loader})


def _sharded_parameter(size: int, weight_loader, *trailing: int) -> nn.Parameter:
    """An empty parameter of `size` rows (on the current default device and dtype,
    so `init_parameters` need not materialise it) loaded by `weight_loader`."""
    param = nn.Parameter(torch.empty(size, *trailing), requires_grad=False)
    set_weight_attrs(param, {"weight_loader": weight_loader})
    return param


def shard_mamba2_mixer(
    mixer: nn.Module, prefix: str, vllm_config: "VllmConfig"
) -> None:
    """Shard a Transformers Mamba2 mixer for tensor parallelism, in place.

    Its projections, convolution, SSM parameters and gated norm are replaced with
    sharded equivalents, laid out as in `MambaMixer2` (including replicating a single
    group to every rank). Its size attributes are then set to their rank-local
    values, so that its forward, which reads them to split its projections, splits
    this rank's shard. The global sizes are kept in `VLLM_SSD_SIZES_ATTR` for
    `TransformersMamba2SSM`.
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    quant_config = vllm_config.quant_config
    intermediate_size = mixer.intermediate_size
    n_groups = mixer.n_groups
    num_heads = mixer.num_heads
    state_size = mixer.ssm_state_size

    if num_heads % tp_size != 0:
        raise NotImplementedError(
            f"{prefix} has {num_heads} heads, which tensor parallel size {tp_size} "
            "does not divide."
        )
    if n_groups % tp_size != 0 and n_groups != 1:
        raise NotImplementedError(
            f"{prefix} has {n_groups} groups, which tensor parallel size {tp_size} "
            "does not divide. Only a single group can be replicated across ranks."
        )
    in_proj, out_proj = mixer.in_proj, mixer.out_proj
    projection_size = intermediate_size + mixer.conv_dim + num_heads
    if not (
        isinstance(in_proj, nn.Linear)
        and isinstance(out_proj, nn.Linear)
        and in_proj.out_features == projection_size
    ):
        raise NotImplementedError(
            f"{prefix} does not project to (gate, x, B, C, dt) with `nn.Linear`s."
        )

    # If tensor parallel size does not divide the groups, replicate them so that
    # each head shard has its group (see `MambaMixer2`)
    extra_groups = 0
    if n_groups % tp_size != 0:
        extra_groups = MambaStateShapeCalculator.extra_groups_for_head_shards(
            n_groups, tp_size
        )
    groups_size = (n_groups + extra_groups) * state_size
    # (global size, extra size from replication, whether groups are replicated)
    gate_or_x = (intermediate_size, 0, False)
    group = (groups_size, extra_groups * state_size, n_groups == 1)
    head = (num_heads, 0, False)
    conv_loader = mamba_v2_sharded_weight_loader(
        [gate_or_x, group, group], tp_size, tp_rank
    )
    in_proj_loader = mamba_v2_sharded_weight_loader(
        [gate_or_x, gate_or_x, group, group, head], tp_size, tp_rank
    )
    head_loader = sharded_weight_loader(0)

    # 1. Projections
    mixer.in_proj = ColumnParallelLinear(
        input_size=in_proj.in_features,
        output_size=2 * intermediate_size + 2 * groups_size + num_heads,
        bias=in_proj.bias is not None,
        quant_config=quant_config,
        prefix=f"{prefix}.in_proj",
        return_bias=False,
    )
    _set_weight_loader(mixer.in_proj.weight, in_proj_loader)
    if mixer.in_proj.bias is not None:
        _set_weight_loader(mixer.in_proj.bias, in_proj_loader)
    mixer.out_proj = RowParallelLinear(
        input_size=intermediate_size,
        output_size=out_proj.out_features,
        bias=out_proj.bias is not None,
        input_is_parallel=True,
        quant_config=quant_config,
        prefix=f"{prefix}.out_proj",
        return_bias=False,
    )
    # FalconH1 scales the projection by a vector as wide as it
    if isinstance(mup_vector := getattr(mixer, "mup_vector", None), torch.Tensor):
        local_mup_vector = torch.empty(
            mixer.in_proj.output_size_per_partition, dtype=mup_vector.dtype
        )
        in_proj_loader(local_mup_vector, mup_vector.reshape(-1))
        mixer.mup_vector = local_mup_vector.view(*mup_vector.shape[:-1], -1)

    # 2. Convolution, of which only the weights are used
    conv1d = mixer.conv1d
    conv_size = (intermediate_size + 2 * groups_size) // tp_size
    conv1d.weight = _sharded_parameter(conv_size, conv_loader, *conv1d.weight.shape[1:])
    if conv1d.bias is not None:
        conv1d.bias = _sharded_parameter(conv_size, conv_loader)

    # 3. SSM parameters, sharded by head
    for name in ("A_log", "D", "dt_bias"):
        setattr(mixer, name, _sharded_parameter(num_heads // tp_size, head_loader))

    # 4. Gated norm, which may reduce over all ranks
    norm = mixer.norm
    if getattr(norm, "norm_before_gate", False):
        # `Mixer2RMSNormGated` gates before normalising
        if tp_size > 1:
            raise NotImplementedError(
                f"{prefix} normalises before gating, which is not supported with "
                "tensor parallelism."
            )
    elif isinstance(getattr(norm, "weight", None), nn.Parameter):
        # The norm normalises over groups of `group_size`, its `n_groups` groups,
        # or the whole of `intermediate_size`
        if group_size := getattr(norm, "group_size", None):
            norm_groups = intermediate_size // group_size
        else:
            norm_groups = getattr(norm, "n_groups", 1)
        mixer.norm = Mixer2RMSNormGated(
            intermediate_size, norm_groups, eps=norm.variance_epsilon
        )

    # 5. Sizes, made rank-local for the forward
    setattr(
        mixer, VLLM_SSD_SIZES_ATTR, Mamba2Sizes(intermediate_size, n_groups, num_heads)
    )
    mixer.intermediate_size = intermediate_size // tp_size
    mixer.num_heads = num_heads // tp_size
    mixer.n_groups = (n_groups + extra_groups) // tp_size
    mixer.conv_dim = mixer.intermediate_size + 2 * mixer.n_groups * state_size


class TransformersMamba2SSM(Mamba2SSM):
    """`Mamba2SSM` that reads its weights from the Transformers mixer it serves."""

    def __init__(self, mixer: nn.Module, vllm_config: "VllmConfig", prefix: str):
        sizes: Mamba2Sizes = getattr(mixer, VLLM_SSD_SIZES_ATTR)
        super().__init__(
            ssm_state_size=mixer.ssm_state_size,
            conv_kernel_size=mixer.conv_kernel_size,
            intermediate_size=sizes.intermediate_size,
            n_groups=sizes.n_groups,
            num_heads=sizes.num_heads,
            head_dim=mixer.head_dim,
            activation=mixer.activation,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            prefix=prefix,
        )
        # Weak, so the mixer does not become a submodule of its own child
        self._mixer = weakref.ref(mixer)
        A = torch.empty(sizes.num_heads // self.tp_size, dtype=torch.float32)
        self.register_buffer("A", A, persistent=False)

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        mixer = self.mixer
        self.A.copy_(-torch.exp(mixer.A_log.float()))
        # Views of the mixer's loaded weights, so they are not parameters of this
        # module but still see in-place weight updates
        weight = mixer.conv1d.weight.data
        self.conv_weights = weight.view(weight.size(0), weight.size(2))
        self.D = mixer.D.data
        self.dt_bias = mixer.dt_bias.data

    @property
    def mixer(self) -> nn.Module:
        mixer = self._mixer()
        assert mixer is not None
        return mixer

    @property
    def conv_bias(self) -> torch.Tensor | None:
        return self.mixer.conv1d.bias


class LinearAttentionMixin(HasInnerState, SupportsMambaPrefixCaching):
    """State interfaces for models whose linear attention layers are all SSD.

    The geometry is read from the config before the model exists, so the
    attribute names cover the Mamba2 family's configs.
    """

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        num_heads = getattr_iter(
            config, ("mamba_n_heads", "mamba_num_heads", "n_mamba_heads", "num_heads")
        )
        head_dim = getattr_iter(
            config, ("mamba_d_head", "mamba_head_dim", "mamba_headdim", "head_dim")
        )
        return MambaStateShapeCalculator.mamba2_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=num_heads * head_dim,
            n_groups=getattr_iter(
                config, ("mamba_n_groups", "mamba_ngroups", "n_groups")
            ),
            num_heads=num_heads,
            head_dim=head_dim,
            state_size=getattr_iter(
                config, ("mamba_d_state", "ssm_state_size", "state_size")
            ),
            conv_kernel=getattr_iter(config, ("mamba_d_conv", "conv_kernel")),
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()
