# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8Dynamic128Sym,
)
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
    _fx_view_to_reshape,
    fold_consecutive_reshapes,
)
from .matcher_utils import (
    MatcherQuantFP8,
    MatcherRMSNormGated,
    MatcherSiluAndMul,
)
from .rms_quant_fusion import (
    FusedRMSQuantKey,
)

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()


class AiterRMSNormQuantPattern:
    def __init__(
        self, epsilon: float, key: FusedRMSQuantKey, match_aiter_quant: bool = True
    ):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype
        self.device = torch.device("cuda")

        self.quant_matcher = MatcherQuantFP8(
            key.quant,
            match_rocm_aiter=match_aiter_quant,
        )

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.bfloat16, device=self.device, **kwargs)

    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kwargs)


class AiterRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = torch.ops.vllm_ir.rms_norm(input, weight, self.epsilon)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1]

        pm.register_replacement(
            pattern,
            replacement,
            # input, weight
            [self.empty(5, 16), self.empty(16)],
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm Fused Add + Dynamic Quantization pattern."""

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_fused_add_dynamic_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        match_aiter_quant: bool = True,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
                quant_dtype=self.quant_dtype,
            )

            return result[0], result[1], result[2]

        inputs = [
            self.empty(5, 16),  # input
            self.empty(16),  # weight
            self.empty(5, 16),  # residual
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class AiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm & group fp8 quant custom
    ops into an aiter rms_norm_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = torch.ops.vllm_ir.rms_norm(input, weight, self.epsilon)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            return at[0], at[1]

        pm.register_replacement(
            pattern,
            replacement,
            # input, weight
            [self.empty(5, 16), self.empty(16)],
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm_with_add & group fp8 quant custom ops
    into a aiter rms_norm_with_add_group_fp8_quant op.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_add_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            result, scale = self.quant_matcher(result_rms)

            return result, residual_out, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = self.FUSED_OP(
                x=input,
                residual=residual,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            # result, scale, residual
            return at[0], at[1], at[2]

        inputs = [
            self.empty(5, 16),  # input
            self.empty(16),  # weight
            self.empty(5, 16),  # residual
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)


class DoubleAiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    Pattern matching ``rms_norm`` whose output feeds *two* distinct
    ``rocm_aiter_group_fp8_quant`` consumers, replacing it with two
    independent fused ``rms_norm_group_fp8_quant`` ops.

    Repeating the rms_norm in the replacement is preferable to leaving
    the fused 16-bit rms output materialized for two unfused quant
    consumers, and matches what the previous manual graph surgery
    achieved by cloning the rms_norm node.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms = torch.ops.vllm_ir.rms_norm(input, weight, self.epsilon)
            result1, scale1 = self.quant_matcher(result_rms)
            result2, scale2 = self.quant_matcher(result_rms)
            return result1, scale1, result2, scale2

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            at1 = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )
            at2 = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            return at1[0], at1[1], at2[0], at2[1]

        pm.register_replacement(
            pattern,
            replacement,
            # input, weight
            [self.empty(5, 16), self.empty(16)],
            pm.fwd_only,
            pm_pass,
        )


class DoubleAiterRMSFp8GroupQuantViewPattern(AiterRMSNormQuantPattern):
    """
    View-tolerant variant of ``DoubleAiterRMSFp8GroupQuantPattern``.

    Matches the same 1-to-2 fan-out, but with a ``view``/``reshape`` between
    the ``rms_norm`` output and the two ``rocm_aiter_group_fp8_quant``
    consumers::

        rms_norm -> view -> rocm_aiter_group_fp8_quant
                \\-> view -> rocm_aiter_group_fp8_quant

    This shape arises in DeepSeek-V3.2's MLA indexer q_c norm, where the
    FP8 linear path's 2D-flatten boilerplate
    (``Fp8BlockScaledMMLinearKernel.apply_weights``) inserts a view between
    the rms_norm output and each FP8 group quant op. The non-view sibling
    pattern silently no-ops on this graph because the pattern matcher
    requires the in-graph and in-pattern node shapes to align.

    The trace_fn runs Inductor's ``view_to_reshape`` post-grad pass to
    normalize ``view`` to ``reshape`` in both the pattern and the input
    graph, widening the match without touching the no-view sibling.
    """

    FUSED_OP = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        super().__init__(epsilon, key, match_aiter_quant)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms = torch.ops.vllm_ir.rms_norm(input, weight, self.epsilon)
            view_rms = result_rms.view(-1, result_rms.shape[-1])
            result1, scale1 = self.quant_matcher(view_rms)
            result2, scale2 = self.quant_matcher(view_rms)
            return result1, scale1, result2, scale2

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            at1 = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )
            at2 = self.FUSED_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            return at1[0], at1[1], at2[0], at2[1]

        def trace_with_view_to_reshape(*args: Any, **kwargs: Any) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            # input, weight
            [self.empty(5, 16), self.empty(16)],
            trace_with_view_to_reshape,
            pm_pass,
        )


class AiterRMSNormGatedFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    Matches decomposed RMSNormGated + reshape + group FP8 quant and replaces
    with rocm_aiter_fused_rms_gated_fp8_group_quant.

    The norm operates per-head on (N*H, D) tensors. The compiler folds the
    reshape chain so after norm the result goes through reshape->merge->quant.
    The pattern reshapes from (N*H, D) to (N, H*D) before calling
    MatcherQuantFP8 so that _quantize_group_native sees the full hidden dim
    and computes the correct num_groups.
    """

    FUSED_OP = rocm_aiter_ops.get_fused_rms_gated_fp8_group_quant_op()

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        num_heads: int,
        head_dim: int,
        match_aiter_quant: bool = True,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        super().__init__(epsilon, key, match_aiter_quant)
        self.rmsnorm_gated_matcher = MatcherRMSNormGated(epsilon)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def register(self, pm_pass: PatternMatcherPass) -> None:
        num_heads = self.num_heads
        head_dim = self.head_dim
        hidden_dim = num_heads * head_dim
        quant_matcher = self.quant_matcher

        def pattern(
            x: torch.Tensor,
            z: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            normed = self.rmsnorm_gated_matcher(x, z, weight)
            merged = normed.reshape(-1, hidden_dim)
            quant_out, scales_out = quant_matcher(merged)
            return quant_out, scales_out

        def replacement(
            x: torch.Tensor,
            z: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            fused = self.FUSED_OP(
                x=x,
                weight=weight,
                bias=None,
                z=z,
                eps=self.epsilon,
                norm_before_gate=True,
                activation="silu",
                group_size=head_dim,
            )
            fp8_out = fused[0]
            scales_out = fused[1]
            fp8_reshaped = fp8_out.reshape(-1, hidden_dim)
            scales_reshaped = scales_out.reshape(-1, num_heads)
            return fp8_reshaped, scales_reshaped

        n_tokens = 2
        x = self.empty(n_tokens * num_heads, head_dim)
        z = self.empty(n_tokens * num_heads, head_dim)
        w = self.empty(head_dim)

        def trace_fn(*args, **kwargs):
            gm = pm.fwd_only(*args, **kwargs)
            _fx_view_to_reshape(gm)
            fold_consecutive_reshapes(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            [x, z, w],
            trace_fn,
            pm_pass,
        )


class RocmAiterRMSNormQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses aiter rms_norm & vllm/aiter quant custom ops
    into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_rms_norm_quant_fusion_pass"
        )

        # Discover (num_heads, head_dim) pairs for gated RMSNorm patterns
        # from GatedDeltaNetAttention layers in static_forward_context.
        from vllm.model_executor.layers.mamba.gdn_linear_attn import (
            GatedDeltaNetAttention,
        )

        gdn_layers = get_layers_from_vllm_config(config, GatedDeltaNetAttention)
        gated_norm_shapes: set[tuple[int, int]] = set()
        for layer in gdn_layers.values():
            gated_norm_shapes.add(
                (layer.num_v_heads // layer.tp_size, layer.head_v_dim)
            )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops.
        # The DoubleQuant patterns handle 1 rms_norm -> 2 group_fp8_quant
        # fan-out (e.g. DSv3.2) and must be registered before the single
        # group-quant pattern so they match first. The view-tolerant variant
        # additionally covers the rms_norm -> view -> 2x quant shape that
        # appears when the FP8 linear path inserts a 2D-flatten boilerplate
        # (DSv3.2 MLA indexer q_c norm).
        for epsilon in [1e-5, 1e-6]:
            # Fuse aiter rms_norm + 2x aiter group fp8 quant
            DoubleAiterRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            # View-tolerant sibling for DSv3.2 q_c norm fan-out
            DoubleAiterRMSFp8GroupQuantViewPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            #  Fuse aiter rms_norm + aiter dynamic group fp8 quant
            AiterRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            # Fuse aiter fused_add_rms_norm + aiter dynamic group fp8 quant
            AiterFusedAddRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128)
            ).register(self.patterns)

            # When quant_fp8 custom ops are disabled, both AITER and native
            # quant matchers trace through QuantFP8's native implementation.
            # Registering both variants would create duplicate Inductor
            # patterns.
            is_quant_fp8_enabled = config.compilation_config.is_custom_op_enabled(
                "quant_fp8"
            )
            match_aiter_quant_options = (
                [True, False] if is_quant_fp8_enabled else [False]
            )

            for match_aiter_quant in match_aiter_quant_options:
                # Fuse aiter rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

                # Fuse aiter fused_add_rms_norm + (aiter / vllm built-in)
                # dynamic per-token fp8 quant
                AiterFusedAddRMSNormDynamicQuantPattern(
                    epsilon, FP8_DTYPE, match_aiter_quant=match_aiter_quant
                ).register(self.patterns)

            # Fuse decomposed RMSNormGated + group fp8 quant.
            # The replacement op (fused_rms_gated_fp8_group_quant) requires
            # an aiter version that includes the GDN triton kernel renames.
            if gated_norm_shapes and rocm_aiter_ops.are_gdn_triton_kernels_available():
                for num_heads, head_dim in gated_norm_shapes:
                    if head_dim != 128:
                        continue
                    AiterRMSNormGatedFp8GroupQuantPattern(
                        epsilon,
                        FP8_DTYPE,
                        GroupShape(1, 128),
                        num_heads=num_heads,
                        head_dim=head_dim,
                    ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug(
            "%s Replaced %s patterns", self.__class__.__name__, self.matched_count
        )

    def uuid(self) -> str:
        fusion_patterns = [
            AiterRMSNormDynamicQuantPattern,
            AiterFusedAddRMSNormDynamicQuantPattern,
            AiterRMSFp8GroupQuantPattern,
            AiterFusedAddRMSFp8GroupQuantPattern,
            DoubleAiterRMSFp8GroupQuantPattern,
            DoubleAiterRMSFp8GroupQuantViewPattern,
            AiterRMSNormGatedFp8GroupQuantPattern,
        ]
        return self.hash_source(self, *fusion_patterns)


class AiterSiluMulFp8GroupQuantPattern(VllmPatternReplacement):
    """
    This pattern fuses aiter silu_and_mul & group fp8 quant custom
    ops into an aiter silu_and_mul_group_fp8_quant op.
    """

    FUSED_SILU_MUL_QUANT_OP = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()

    def __init__(self) -> None:
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.quant_matcher = MatcherQuantFP8(
            quant_key=kFp8Dynamic128Sym, match_rocm_aiter=True
        )

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.silu_and_mul_matcher.inputs()[0],
        ]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = self.silu_and_mul_matcher(input)
            at2 = self.quant_matcher(at1)
            return at2[0], at2[1]

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at = self.FUSED_SILU_MUL_QUANT_OP(x=input, group_size=128)
            return at[0], at[1]

        return _replacement


class RocmAiterSiluMulFp8GroupQuantFusionPass(VllmFusionPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "rocm_aiter_silu_mul_fp8_group_quant_fusion_pass")

        self.register(AiterSiluMulFp8GroupQuantPattern())

        self.dump_patterns(config, self.pm_pass)


class AddAiterRMSNormPadPattern:
    """
    This pattern replaces an aiter_rmsnorm_with_add & a pad op
    with a custom triton_add_rmsnorm_pad op from AITER.
    """

    AITER_TRITON_ADD_RMSNORM_PAD_OP = rocm_aiter_ops.get_triton_add_rmsnorm_pad_op()

    def __init__(
        self,
        epsilon: float,
        hidden_size: int,
        x_pad_to_multiple: int,
    ):
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        self.x_pad_to_multiple = x_pad_to_multiple

    def get_inputs(self) -> list[torch.Tensor]:
        device = torch.device("cuda")
        dtype = torch.bfloat16
        input = torch.empty(5, 16, dtype=dtype, device=device)
        weight = torch.empty(16, dtype=dtype, device=device)
        residual = torch.empty(5, 16, dtype=dtype, device=device)
        router_weight = torch.empty([8, 16], dtype=dtype, device=device)
        router_bias = torch.empty([8], dtype=dtype, device=device)
        return [input, weight, residual, router_weight, router_bias]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            router_weight: torch.Tensor,
            router_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pad_size = self.x_pad_to_multiple - (
                self.hidden_size % self.x_pad_to_multiple
            )
            result_rms, residual_out = torch.ops.vllm_ir.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            router_logits = torch.ops.vllm.rocm_unquantized_gemm(
                result_rms, router_weight, router_bias
            )
            result = torch.nn.functional.pad(
                result_rms, (0, pad_size), mode="constant", value=0.0
            )
            return result, residual_out, router_logits

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            router_weight: torch.Tensor,
            router_bias: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            at = self.AITER_TRITON_ADD_RMSNORM_PAD_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                residual=residual,
                x_pad_to_multiple=self.x_pad_to_multiple,
            )
            result_padded = at[0]
            router_logits = torch.ops.vllm.rocm_unquantized_gemm(
                result_padded[:, : self.hidden_size], router_weight, router_bias
            )
            residual_out = at[1]
            return result_padded, residual_out, router_logits

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class RocmAiterTritonAddRMSNormPadFusionPass(VllmPatternMatcherPass):
    """
    This pass replaces an AITER CK RMSNorm + residual add and a pad op
    with an triton_add_rmsnorm_pad op from AITER.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_triton_add_rmsnorm_pad_fusion_pass"
        )

        # gpt-oss has hidden size 2880
        # padded to a multiple of 128 on gfx942 and 256 on gfx950 respectively
        hidden_size = 2880
        for epsilon in [1e-5, 1e-6]:
            for x_pad_to_multiple in [128, 256]:
                AddAiterRMSNormPadPattern(
                    epsilon, hidden_size, x_pad_to_multiple
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, AddAiterRMSNormPadPattern)


class MLADualRMSNormPattern(
    VllmPatternReplacement[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Fuse paired q_a_layernorm + kv_a_layernorm in MLA attention into
    AITER's ``fused_qk_rmsnorm`` HIP kernel.

    Target FX-graph pattern (unfused, ``vllm_ir`` stage)::

        gemm -> split_with_sizes([q_dim, kv_dim])
            +-- q_c     -> vllm_ir.rms_norm(q_c, q_w, eps)
            +-- kv_lora -> split_with_sizes([kv_c_dim, k_pe_dim])
                            +-- kv_c -> vllm_ir.rms_norm(kv_c, kv_w, eps)
                            +-- k_pe

    The pattern covers the connected subgraph rooted at the first
    ``split_with_sizes`` (which produces ``q_c`` and ``kv_lora``),
    through the two ``rms_norm`` calls, and the ``k_pe`` passthrough.
    """

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def get_inputs(self) -> list[torch.Tensor]:
        q_dim, kv_c_dim, k_pe_dim = 8, 4, 2
        return [
            self.empty_bf16(5, q_dim + kv_c_dim + k_pe_dim),
            self.empty_bf16(q_dim),
            self.empty_bf16(kv_c_dim),
        ]

    @property
    def pattern(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        eps = self._epsilon

        def _pattern(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_dim = q_weight.shape[0]
            kv_dim = projected.shape[-1] - q_dim
            kv_c_dim = kv_weight.shape[0]
            k_pe_dim = kv_dim - kv_c_dim
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            q_normed = vllm.ir.ops.rms_norm(q_c, q_weight, eps)
            kv_normed = vllm.ir.ops.rms_norm(kv_c, kv_weight, eps)
            return q_normed, kv_normed, k_pe

        return _pattern

    @property
    def replacement(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        eps = self._epsilon

        def _replacement(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_dim = q_weight.shape[0]
            kv_dim = projected.shape[-1] - q_dim
            kv_c_dim = kv_weight.shape[0]
            k_pe_dim = kv_dim - kv_c_dim
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            q_normed, kv_normed = torch.ops.vllm.fused_mla_dual_rms_norm(
                q_c,
                q_weight,
                kv_c,
                kv_weight,
                eps,
                eps,
            )
            return q_normed, kv_normed, k_pe

        return _replacement


class MLADualRMSNormFusionPass(VllmFusionPatternMatcherPass):
    """
    Post-grad PatternMatcher pass that fuses paired q / kv RMS norms in
    MLA attention into ``fused_mla_dual_rms_norm`` backed by aiter's
    ``fused_qk_rmsnorm`` HIP kernel.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "mla_dual_rms_norm_fusion_pass")

        for epsilon in [1e-5, 1e-6]:
            self.register(MLADualRMSNormPattern(epsilon))
