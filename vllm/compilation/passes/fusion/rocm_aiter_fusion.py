# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import operator
import os
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.envs as envs
import vllm.ir.ops
import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
from vllm._aiter_ops import (
    rocm_aiter_ops,
    rocm_aiter_rmsnorm_mxfp4_quant_args_error,
)
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8Dynamic128Sym,
)
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode, get_pass_context
from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
    _fx_view_to_reshape,
    fold_consecutive_reshapes,
    remove_noop_reshapes,
)
from .matcher_utils import (
    MatcherGeluAndMul,
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
    """This pattern fuses aiter rms_norm & group fp8 quant custom
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
    """This pattern fuses aiter rms_norm_with_add & group fp8 quant custom ops
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
    """Pattern matching ``rms_norm`` whose output feeds *two* distinct
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
    """View-tolerant variant of ``DoubleAiterRMSFp8GroupQuantPattern``.

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
    """Matches decomposed RMSNormGated + reshape + group FP8 quant and replaces
    with rocm_aiter_fused_rms_gated_fp8_group_quant.

    The norm operates per-head, on either (N*H, D) or (N, H, D). The compiler
    folds the reshape chain so after norm the result goes through
    reshape->merge->quant. The pattern reshapes to (N, H*D) before calling
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
        self._register_layout(pm_pass, flatten_heads=True)
        # Both layouts reach the norm. With an opaque quant op they trace to
        # the same pattern; with a native quant they differ, because its group
        # reshape back to (N, H, D) is a no-op the compiler already dropped.
        if not self.quant_matcher.enabled:
            self._register_layout(pm_pass, flatten_heads=False)

    def _register_layout(
        self, pm_pass: PatternMatcherPass, *, flatten_heads: bool
    ) -> None:
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
            # The pattern matches on op structure rather than rank, so either
            # per-head layout can land here. The fused op takes (M, D), and
            # both operands are contiguous, so collapsing the heads is a view.
            x = x.reshape(-1, head_dim)
            z = z.reshape(-1, head_dim)
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
        shape = (
            (n_tokens * num_heads, head_dim)
            if flatten_heads
            else (n_tokens, num_heads, head_dim)
        )
        x = self.empty(*shape)
        z = self.empty(*shape)
        w = self.empty(head_dim)

        def trace_fn(*args, **kwargs):
            gm = pm.fwd_only(*args, **kwargs)
            _fx_view_to_reshape(gm)
            fold_consecutive_reshapes(gm)
            if not flatten_heads:
                remove_noop_reshapes(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            [x, z, w],
            trace_fn,
            pm_pass,
        )


class RocmAiterRMSNormQuantFusionPass(VllmPatternMatcherPass):
    """This pass fuses aiter rms_norm & vllm/aiter quant custom ops
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
        from vllm.model_executor.layers.mamba.gdn.base import (
            GatedDeltaNetAttention,
        )

        gdn_layers = get_layers_from_vllm_config(
            config,
            GatedDeltaNetAttention,  # type: ignore[type-abstract]
        )
        gated_norm_shapes: set[tuple[int, int]] = set()
        for layer in gdn_layers.values():
            num_v_heads = getattr(layer, "num_v_heads", None) or getattr(
                layer, "num_heads", None
            )
            head_v_dim = getattr(layer, "head_v_dim", None) or getattr(
                layer, "head_dim", None
            )

            assert num_v_heads is not None and head_v_dim is not None

            gated_norm_shapes.add((num_v_heads // layer.tp_size, head_v_dim))

        # RDNA4 uses native quant ops and supports only Triton replacements.
        match_aiter_quant_op = not rocm_aiter_ops.is_rdna_aiter_enabled()

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
                epsilon, FP8_DTYPE, GroupShape(1, 128), match_aiter_quant_op
            ).register(self.patterns)

            # View-tolerant sibling for DSv3.2 q_c norm fan-out
            DoubleAiterRMSFp8GroupQuantViewPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128), match_aiter_quant_op
            ).register(self.patterns)

            #  Fuse aiter rms_norm + aiter dynamic group fp8 quant
            AiterRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128), match_aiter_quant_op
            ).register(self.patterns)

            # Fuse aiter fused_add_rms_norm + aiter dynamic group fp8 quant
            AiterFusedAddRMSFp8GroupQuantPattern(
                epsilon, FP8_DTYPE, GroupShape(1, 128), match_aiter_quant_op
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
            if not match_aiter_quant_op:
                match_aiter_quant_options = []

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
            if gated_norm_shapes and (
                rocm_aiter_ops.are_gdn_triton_kernels_available()
                or rocm_aiter_ops.is_rdna_gdn_triton_kernels_available()
            ):
                for num_heads, head_dim in gated_norm_shapes:
                    if head_dim != 128:
                        continue
                    AiterRMSNormGatedFp8GroupQuantPattern(
                        epsilon,
                        FP8_DTYPE,
                        GroupShape(1, 128),
                        num_heads=num_heads,
                        head_dim=head_dim,
                        match_aiter_quant=match_aiter_quant_op,
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
    """This pattern fuses aiter silu_and_mul & group fp8 quant custom
    ops into an aiter silu_and_mul_group_fp8_quant op.
    """

    FUSED_SILU_MUL_QUANT_OP = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()

    def __init__(self, match_aiter_quant_op: bool = True) -> None:
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.quant_matcher = MatcherQuantFP8(
            quant_key=kFp8Dynamic128Sym, match_rocm_aiter=match_aiter_quant_op
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
    """This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "rocm_aiter_silu_mul_fp8_group_quant_fusion_pass")

        match_aiter_quant_op = not rocm_aiter_ops.is_rdna_aiter_enabled()

        self.register(
            AiterSiluMulFp8GroupQuantPattern(match_aiter_quant_op=match_aiter_quant_op)
        )

        self.dump_patterns(config, self.pm_pass)


class AiterActMulMxfp4GemmPattern(VllmPatternReplacement):
    """Folds act_and_mul into the AITER MXFP4 GEMM that consumes it.

    The ASM MXFP4 linear quantizes its input inside gemm_with_dynamic_quant, a
    custom op Inductor cannot fuse into. This pattern hands the op the
    concatenated gate/up tensor plus the activation name instead, and the op
    runs aiter's act_mul_and_mxfp4_quant: one kernel instead of an activation
    kernel plus a quant kernel.
    """

    def __init__(
        self,
        act_matcher: MatcherGeluAndMul | MatcherSiluAndMul,
        activation: str,
        out_dtype: torch.dtype,
    ) -> None:
        # gemm_with_dynamic_quant is registered when the MXFP4 kernel module
        # is imported, so resolve it here rather than at class definition.
        import vllm.model_executor.kernels.linear.mxfp4.aiter  # noqa: F401

        self.gemm_op = torch.ops.vllm.gemm_with_dynamic_quant.default
        self.act_matcher = act_matcher
        self.activation = activation
        self.out_dtype = out_dtype

    def get_inputs(self) -> list[torch.Tensor]:
        x = self.act_matcher.inputs()[0]
        weight = self.empty(8, 1, dtype=torch.uint8)
        weight_scale = self.empty(8, 1, dtype=torch.uint8)
        return [x, weight, weight_scale]

    @property
    def pattern(self):
        def _pattern(
            x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
        ) -> torch.Tensor:
            y = self.act_matcher(x)
            return self.gemm_op(y, weight, weight_scale, True, self.out_dtype)

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
        ) -> torch.Tensor:
            return self.gemm_op(
                x, weight, weight_scale, True, self.out_dtype, None, self.activation
            )

        return _replacement


def _model_dtype(config: VllmConfig) -> torch.dtype:
    """The dtype of the model's activations (bf16 when there is no model config)."""
    if config.model_config is None:
        return torch.bfloat16
    dtype = config.model_config.dtype
    assert isinstance(dtype, torch.dtype), dtype
    return dtype


# The compile prefix of a graph whose model tag was not set (`model_tag` in
# compilation/backends.py).
DEFAULT_COMPILE_PREFIX = "backbone"

# Compile prefixes (VllmBackend.prefix) of the graphs that belong to speculative
# decoding rather than to the target model, taken from the set_model_tag(...) calls
# in v1/spec_decode, v1/worker/gpu/spec_decode and the draft parts of
# models/qwen3_dflash2.py and models/lilicorr.py. The MTP head is bf16 and has no
# MXFP4 GEMM to fuse into, so a fusion pass finding no site in these graphs is
# expected. tests/compile/passes/test_rocm_aiter_mxfp4_fusion_gating.py checks this
# set against every set_model_tag literal in the source tree.
DRAFT_GRAPH_PREFIXES = frozenset(
    {
        "eagle_head",
        "draft_model",
        "medusa_head",
        "dflash_head",
        "dspark_head",
        "extract_hidden_states",
        "dflash2_candidate_selector",
        "lilicorr_head",
    }
)


def is_target_model_graph(prefix: str) -> bool:
    """Whether a compile prefix names a graph of the target model.

    False for the graphs of speculative decoding (DRAFT_GRAPH_PREFIXES), True for
    everything else, including the model-internal tags such as ``self_decoder``.
    """
    return prefix not in DRAFT_GRAPH_PREFIXES


def _mxfp4_pass_gate(fuse_in_pass_config: bool, opt_in_flag: bool) -> bool:
    """The rule the MXFP4 activation-quant fusion passes share.

    The pass config asks for the fusion AND the opt-in env flag is set AND the GPU
    is a gfx950, the only architecture with the asm MXFP4 GEMM these passes rewrite
    the input of. The default optimization level turns the pass config switches
    (``fuse_act_quant``, ``fuse_norm_quant``) on, so the env flag is what keeps a
    default start unchanged.
    """
    from vllm.platforms.rocm import on_gfx950

    return bool(fuse_in_pass_config) and opt_in_flag and on_gfx950()


def act_mxfp4_pass_enabled(fuse_act_quant: bool) -> bool:
    """Whether RocmAiterActMulMxfp4GemmFusionPass is registered.

    Needs ``fuse_act_quant`` and VLLM_ROCM_AITER_FP4_ACT_QUANT_FUSION on a gfx950.
    """
    return _mxfp4_pass_gate(fuse_act_quant, envs.VLLM_ROCM_AITER_FP4_ACT_QUANT_FUSION)


class TargetGraphSiteCheckMixin:
    """Counts the rewrites of a fusion pass and checks the target model got some.

    Put it before VllmFusionPatternMatcherPass in the bases and set two attributes:
    ``flag_name``, the env flag that switches the pass on (quoted in the error), and
    ``site_label``, what is fused (as the log line says it). They are class attributes,
    or instance attributes set before ``super().__init__`` when one class serves
    several flags (RocmAiterRMSNormMxfp4GemmFusionPass has one instance per flag).

    vLLM calls a pass once per compiled piece and per compile range, and a piece such
    as the first one (embedding, layer 0) legitimately has no site, so ``__call__``
    only tallies the rewrites and never raises. ``finish_compile`` judges the tally;
    VllmBackend calls it, through PostGradPassManager.finish_compile, after the last
    piece. N in the log line is the number of sites in the pieces the pass ran on:
    vLLM compiles identical pieces once, so it can be lower than the layer count.
    """

    flag_name: str
    site_label: str
    # Set by the VllmPatternMatcherPass this mixin is combined with.
    matched_count: int

    def __init__(self, *args, compile_prefix: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The VllmBackend prefix of the graph being compiled: "backbone" for the
        # model, "eagle_head" for an MTP head. It names the graph in the log and
        # decides whether finding no site is an error.
        self.compile_prefix = compile_prefix or DEFAULT_COMPILE_PREFIX
        self.sites_by_range: dict[str, int] = {}

    def __call__(self, graph: torch.fx.Graph) -> None:
        super().__call__(graph)  # type: ignore[misc]
        compile_range = str(get_pass_context().compile_range)
        self.sites_by_range[compile_range] = (
            self.sites_by_range.get(compile_range, 0) + self.matched_count
        )

    def finish_compile(self) -> None:
        """Log the rewritten sites per compile range, and fail a start without any.

        A target model graph fails when the pass ran for a compile range and
        rewrote nothing there: the flag is on, so a start that leaves the model
        unfused should not pass silently. A graph the pass never ran on is only
        warned about: Inductor's FX graph cache can serve every piece without running
        the passes, so nothing can be judged. Draft model graphs are only logged.
        """
        for compile_range, sites in self.sites_by_range.items():
            logger.info(
                "Fused %s at %d sites (graph=%s) for compile range %s",
                self.site_label,
                sites,
                self.compile_prefix,
                compile_range,
            )
        if not is_target_model_graph(self.compile_prefix):
            return
        if not self.sites_by_range:
            cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "torch_compile_cache")
            logger.warning(
                "%s is on but the pass ran on no piece of the %s graph, so nothing "
                "was rewritten or checked: Inductor's cache served every piece. "
                "Clear %s to compile with the pass.",
                self.flag_name,
                self.compile_prefix,
                cache_dir,
            )
            return
        if any(sites == 0 for sites in self.sites_by_range.values()):
            raise RuntimeError(
                f"{self.flag_name} is on but 0 sites were rewritten in the "
                f"{self.compile_prefix} graph for compile range(s) "
                f"{[r for r, n in self.sites_by_range.items() if n == 0]}"
            )


class RocmAiterActMulMxfp4GemmFusionPass(
    TargetGraphSiteCheckMixin, VllmFusionPatternMatcherPass
):
    """Fuses SiLU/GELU-and-mul into the following AITER ASM MXFP4 GEMM."""

    flag_name = "VLLM_ROCM_AITER_FP4_ACT_QUANT_FUSION"
    site_label = "act+mxfp4 quant"

    def __init__(self, config: VllmConfig, compile_prefix: str | None = None) -> None:
        super().__init__(
            config,
            "rocm_aiter_act_mul_mxfp4_gemm_fusion_pass",
            compile_prefix=compile_prefix,
        )
        out_dtype = _model_dtype(config)
        matchers: dict[tuple[bool, str], MatcherGeluAndMul | MatcherSiluAndMul] = {}
        silu = MatcherSiluAndMul()
        matchers[(silu.enabled, "silu")] = silu
        for approximate in ("none", "tanh"):
            gelu = MatcherGeluAndMul(approximate)
            # On ROCm both approximations trace to the same native graph.
            matchers.setdefault((gelu.enabled, gelu.activation), gelu)
        for (_, activation), matcher in matchers.items():
            self.register(AiterActMulMxfp4GemmPattern(matcher, activation, out_dtype))

        self.dump_patterns(config, self.pm_pass)


NormSiteKind = Literal["post", "input"]


class NormSites(NamedTuple):
    """What tells the norm pass of one site kind from the other's."""

    flag_name: str  # the env flag that switches the pass on, quoted in startup errors
    site_label: str  # what is fused, as the startup log line says it
    pass_name: str  # name of the pass's PatternMatcherPass (dumps, match table)


# The two kinds of norm site and their passes, one flag each. A decoder layer has a
# post-attention norm (its output feeds the MLP's gate_up) and an input norm (its output
# feeds the attention or gated delta net projections); see
# AiterAddRMSNormMxfp4GemmPattern._norm_site_kind for how the two are told apart. The
# order is the order of the passes in the pass manager.
NORM_SITES: dict[NormSiteKind, NormSites] = {
    "post": NormSites(
        "VLLM_ROCM_AITER_FP4_POSTNORM_QUANT_FUSION",
        "post-attention add+rmsnorm+mxfp4 quant",
        "rocm_aiter_rmsnorm_mxfp4_gemm_fusion_pass",
    ),
    "input": NormSites(
        "VLLM_ROCM_AITER_FP4_INPUTNORM_QUANT_FUSION",
        "input add+rmsnorm+mxfp4 quant",
        "rocm_aiter_input_rmsnorm_mxfp4_gemm_fusion_pass",
    ),
}


def _norm_sites(sites: NormSiteKind) -> NormSites:
    """The NORM_SITES entry of a site kind; ValueError names the kinds there are."""
    try:
        return NORM_SITES[sites]
    except KeyError:
        raise ValueError(
            f"sites must be one of {sorted(NORM_SITES)}, not {sites!r}"
        ) from None


def norm_mxfp4_pass_enabled(
    fuse_norm_quant: bool, sites: NormSiteKind = "post"
) -> bool:
    """Whether the norm + MXFP4 quant pass of one site kind is registered.

    Needs ``fuse_norm_quant`` and the kind's own flag
    (VLLM_ROCM_AITER_FP4_POSTNORM_QUANT_FUSION or
    VLLM_ROCM_AITER_FP4_INPUTNORM_QUANT_FUSION) on a gfx950; see _mxfp4_pass_gate.

    The caller only asks inside `if fuse_norm_quant`; the parameter keeps this helper
    parallel to act_mxfp4_pass_enabled and safe to call from anywhere.
    """
    return _mxfp4_pass_gate(
        fuse_norm_quant, getattr(envs, _norm_sites(sites).flag_name)
    )


class AiterAddRMSNormMxfp4GemmPattern(VllmPatternReplacement):
    """Folds a Gemma add+RMSNorm into the AITER MXFP4 GEMM that consumes its output.

    The ASM MXFP4 linear quantizes its input inside gemm_with_dynamic_quant, a custom op
    Inductor cannot fuse into. This pattern replaces the IR norm and the GEMM's own
    quantization by rocm_aiter_rmsnorm_mxfp4_quant (one AITER HIP kernel: residual add,
    Gemma norm, MXFP4 quant) and hands the GEMM its codes and scales through x_scales.

    Matches `fused_add_rms_norm(x, residual, weight.float() + 1.0, eps)` followed by ONE
    asm gemm_with_dynamic_quant on its normed output. The replacement passes the RAW
    bf16 weight, because the kernel adds 1 itself, and wires the norm's residual output
    to the op's third result. Only the norms of one site kind are sites, the
    post-attention norms or the input norms (``sites``; see _norm_site_kind).

    The pattern matcher refuses a match that leaves another user of the normed tensor
    behind, so this pattern cannot take a norm whose output feeds two GEMMs:
    AiterAddRMSNormMxfp4GemmPairPattern is the one for that.
    """

    # The pattern arguments that are GEMM weights: extra_check judges each of them by
    # these names, which are also the parameter names of the pattern function.
    gemm_weight_names: tuple[str, ...] = ("gemm_weight",)

    def __init__(
        self,
        epsilon: float,
        out_dtype: torch.dtype,
        tp_size: int,
        sites: NormSiteKind = "post",
    ) -> None:
        _norm_sites(sites)  # refuse a kind that would never match
        # gemm_with_dynamic_quant is registered when the MXFP4 kernel module is
        # imported, so resolve it here rather than at class definition.
        importlib.import_module("vllm.model_executor.kernels.linear.mxfp4.aiter")
        self.gemm_op = torch.ops.vllm.gemm_with_dynamic_quant.default
        self.norm_op = torch.ops.vllm_ir.fused_add_rms_norm.default
        self.norm_quant_op = torch.ops.vllm.rocm_aiter_rmsnorm_mxfp4_quant.default
        self.norm_targets = (self.norm_op, self.norm_quant_op)
        self.epsilon = epsilon
        self.out_dtype = out_dtype
        self.tp_size = tp_size
        self.sites = sites

    def get_inputs(self) -> list[torch.Tensor]:
        # The hidden size is 64, a multiple of 32: the op's fake raises for any other,
        # so vLLM's usual empty(5, 16) example cannot be used. A GEMM weight holds
        # K / 2 packed bytes per row and its scale one byte per 32 values.
        hidden = 64
        inputs = [
            self.empty_bf16(5, hidden),  # x
            self.empty_bf16(5, hidden),  # residual
            self.empty_bf16(hidden),  # norm weight, the raw parameter
        ]
        for _ in self.gemm_weight_names:
            inputs += [
                self.empty(8, hidden // 2, dtype=torch.uint8),  # gemm weight
                self.empty(8, hidden // 32, dtype=torch.uint8),  # gemm weight scale
            ]
        return inputs

    @property
    def pattern(self):
        # The parameter names are an interface: extra_check reads the matched nodes back
        # by these names through match.kwargs. Rename them in both places.
        def _pattern(
            x: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            gemm_weight: torch.Tensor,
            gemm_weight_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # GemmaRMSNorm.forward_native: the weight reaches the IR op as fp32 w + 1.
            y, residual_out = self.norm_op(
                x, residual, weight.float() + 1.0, self.epsilon
            )
            # use_asm_gemm is the literal True: a GEMM on the Triton path never matches.
            out = self.gemm_op(y, gemm_weight, gemm_weight_scale, True, self.out_dtype)
            return out, residual_out

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            x: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            gemm_weight: torch.Tensor,
            gemm_weight_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x_q, x_scales, residual_out = self.norm_quant_op(
                x, residual, weight, self.epsilon
            )
            out = self.gemm_op(
                x_q, gemm_weight, gemm_weight_scale, True, self.out_dtype, x_scales
            )
            return out, residual_out

        return _replacement

    @staticmethod
    def norm_args_supported(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> bool:
        """Whether rocm_aiter_rmsnorm_mxfp4_quant takes these arguments.

        The op's fake raises ValueError for anything else, which would crash the compile
        when the replacement is traced, so the site must be filtered here. The rules are
        the op's own (rocm_aiter_rmsnorm_mxfp4_quant_args_error); this adds only that a
        symbolic hidden size is refused, because testing it would add a shape guard.
        """
        if x.dim() == 2 and not isinstance(x.shape[1], int):
            return False
        return rocm_aiter_rmsnorm_mxfp4_quant_args_error(x, residual, weight) is None

    @staticmethod
    def _norm_site_kind(residual: fx.Node, norm_targets: tuple) -> NormSiteKind:
        """Whether a norm is a post-attention norm ("post") or the next layer's input
        norm ("input"), by where its residual operand comes from.

        The two are the same site by shape (a norm and the GEMMs that read it). The
        input norm takes the residual that the post-attention norm returned, an item of
        a norm in `norm_targets` (the IR norm, or this pass's op once an earlier site
        was rewritten); the post-attention norm takes the residual that came in with the
        piece. The item index is not checked: classifying a residual as "input" only
        leaves a site unfused, it never rewrites a wrong one. Each kind has its own pass
        (RocmAiterRMSNormMxfp4GemmFusionPass ``sites``), and a pattern takes only its
        own kind.

        Needs the graph split at every attention op, so that a post-attention norm comes
        first in its piece; the pass refuses a config without that split.
        """
        producer = residual.args[0] if residual.target is operator.getitem else None
        if isinstance(producer, fx.Node) and producer.target in norm_targets:
            return "input"
        return "post"

    @staticmethod
    def _gemm_weight_is_untuned(gemm_weight: torch.Tensor) -> bool:
        """Whether the GEMM has no tuned Triton preshuffle config for its weight shape.

        With M <= 64 and such a config, gemm_with_dynamic_quant takes a small-M Triton
        branch that wants unshuffled scales, while the fused op writes the asm
        (shuffled) layout. M is unknown at compile time, so every tuned shape is
        refused, also for the larger M that would take the asm branch.
        """
        if gemm_weight.dim() != 2:
            return False
        n, k_bytes = gemm_weight.shape
        if not (isinstance(n, int) and isinstance(k_bytes, int)):
            return False
        return not rocm_aiter_ops.is_triton_gemm_afp4wfp4_presh_ws_tuned(n, k_bytes)

    def extra_check(self, match: pm.Match) -> bool:
        """Whether this match is a site the fused op and the asm GEMMs can take.

        Checked at match time, because the op's fake raises on what it cannot take:
        tensor parallel size 1 (otherwise an all-reduce sits in front of the norm), a
        norm of this pattern's site kind, arguments the op takes, and every GEMM without
        a tuned preshuffle config. One GEMM that fails leaves the whole site unfused.
        """
        if self.tp_size != 1:
            return False
        kwargs = match.kwargs  # keyed by the parameter names of _pattern
        if self._norm_site_kind(kwargs["residual"], self.norm_targets) != self.sites:
            return False
        x, residual, weight = (
            kwargs[name].meta.get("val") for name in ("x", "residual", "weight")
        )
        gemm_weights = [kwargs[name].meta.get("val") for name in self.gemm_weight_names]
        if any(t is None for t in (x, residual, weight, *gemm_weights)):
            return False  # no shapes and dtypes to judge by
        return self.norm_args_supported(x, residual, weight) and all(
            self._gemm_weight_is_untuned(gemm_weight) for gemm_weight in gemm_weights
        )


class AiterAddRMSNormMxfp4GemmPairPattern(AiterAddRMSNormMxfp4GemmPattern):
    """The same fusion for a norm whose output feeds TWO asm GEMMs: one quantization.

    A gated delta net layer's input norm feeds in_proj_qkvz and in_proj_ba. The
    replacement quantizes once and hands the same codes and scales to both GEMMs, so
    the second quantization the GEMM would run is gone as well. The pattern matcher
    needs the number of users of the normed tensor to be exactly the pattern's, so a
    norm with one, three or more consumers, or with a consumer that is not a
    gemm_with_dynamic_quant, is not matched by this pattern: such a site is left whole,
    never fused for some of its consumers.
    """

    gemm_weight_names = ("gemm_weight", "gemm_weight_2")

    @property
    def pattern(self):
        # Parameter names: see AiterAddRMSNormMxfp4GemmPattern.pattern.
        def _pattern(
            x: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            gemm_weight: torch.Tensor,
            gemm_weight_scale: torch.Tensor,
            gemm_weight_2: torch.Tensor,
            gemm_weight_scale_2: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            y, residual_out = self.norm_op(
                x, residual, weight.float() + 1.0, self.epsilon
            )
            out = self.gemm_op(y, gemm_weight, gemm_weight_scale, True, self.out_dtype)
            out_2 = self.gemm_op(
                y, gemm_weight_2, gemm_weight_scale_2, True, self.out_dtype
            )
            return out, out_2, residual_out

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            x: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            gemm_weight: torch.Tensor,
            gemm_weight_scale: torch.Tensor,
            gemm_weight_2: torch.Tensor,
            gemm_weight_scale_2: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_q, x_scales, residual_out = self.norm_quant_op(
                x, residual, weight, self.epsilon
            )
            out = self.gemm_op(
                x_q, gemm_weight, gemm_weight_scale, True, self.out_dtype, x_scales
            )
            out_2 = self.gemm_op(
                x_q, gemm_weight_2, gemm_weight_scale_2, True, self.out_dtype, x_scales
            )
            return out, out_2, residual_out

        return _replacement

    def extra_check(self, match: pm.Match) -> bool:
        """The single pattern's checks, and the two GEMM slots hold different nodes.

        The matcher tries the users of the normed tensor in the order the graph lists
        them for the second slot, so it can bind both slots to the same GEMM (the first
        user, when that is also the anchor it started from). That match would replace
        one GEMM twice and leave the other one reading the tensor that is gone; it is
        refused, and the matcher goes on to the other GEMM as the anchor, which finds
        the real pair.
        """
        outputs = match.output_nodes()
        return len(set(outputs)) == len(outputs) and super().extra_check(match)


class RocmAiterRMSNormMxfp4GemmFusionPass(
    TargetGraphSiteCheckMixin, VllmFusionPatternMatcherPass
):
    """Fuses a Gemma add+RMSNorm into the AITER ASM MXFP4 GEMM(s) after it.

    One pass of one site kind (``sites``, a key of NORM_SITES), with the flag, the log
    label and the pattern matcher name of that kind. The pass manager builds one
    instance per enabled flag, so each flag is judged on its own sites (the mixin) and
    its startup error names its own flag.

    "post": the norm that feeds the MLP's gate_up projection, one site per decoder
    layer, one GEMM each (AiterAddRMSNormMxfp4GemmPattern).
    "input": the norm that starts a decoder layer. Its output feeds one GEMM in a
    full-attention layer (qkv_proj) and two in a gated delta net layer (in_proj_qkvz
    and in_proj_ba); the pass quantizes once and gives the codes and scales to every
    GEMM, by the single pattern or by AiterAddRMSNormMxfp4GemmPairPattern. Layer 0's
    input norm is a plain rms_norm with no residual and is not a site.

    The model graph must be split at the attention ops (see
    AiterAddRMSNormMxfp4GemmPattern._norm_site_kind), which __init__ checks.
    """

    # The epsilon is a constant inside the traced pattern: one pattern per value, the
    # same two the FP8 norm passes register.
    EPSILONS = (1e-5, 1e-6)
    # The patterns of each site kind. A post-attention norm with two GEMMs is not a
    # site of the post kind.
    PATTERNS: dict[NormSiteKind, tuple[type[AiterAddRMSNormMxfp4GemmPattern], ...]] = {
        "post": (AiterAddRMSNormMxfp4GemmPattern,),
        "input": (
            AiterAddRMSNormMxfp4GemmPattern,
            AiterAddRMSNormMxfp4GemmPairPattern,
        ),
    }

    def __init__(
        self,
        config: VllmConfig,
        compile_prefix: str | None = None,
        sites: NormSiteKind = "post",
    ) -> None:
        spec = _norm_sites(sites)
        self.sites = sites
        self.flag_name = spec.flag_name
        self.site_label = spec.site_label
        self._require_split_graph(config)
        super().__init__(config, spec.pass_name, compile_prefix=compile_prefix)
        out_dtype = _model_dtype(config)
        tp_size = config.parallel_config.tensor_parallel_size
        for epsilon in self.EPSILONS:
            for pattern_cls in self.PATTERNS[sites]:
                self.register(pattern_cls(epsilon, out_dtype, tp_size, sites))

        self.dump_patterns(config, self.pm_pass)

    def uuid(self) -> str:
        # Both site kinds are this class and may share pattern classes: the kind is what
        # tells their rewrites, and so their compile cache keys, apart.
        return VllmInductorPass.hash_source(super().uuid(), self.sites)

    def _require_split_graph(self, config: VllmConfig) -> None:
        """Refuse a config under which the site classifier would misjudge every norm.

        The classifier needs the graph split into pieces at the attention ops. With
        Inductor graph partition, or with no splitting ops, the whole model is one
        graph: only the first norm would count as a site, the start-up check would
        pass, and the rest would silently stay unfused.
        """
        compilation = config.compilation_config
        if compilation.use_inductor_graph_partition:
            reason = "use_inductor_graph_partition is on"
        elif not compilation.splitting_ops:
            reason = "compilation_config.splitting_ops is empty"
        else:
            return
        raise RuntimeError(
            f"{self.flag_name} is on but {reason}, so the model graph is not split at "
            "the attention ops and the pass cannot tell a post-attention norm from the "
            "next layer's input norm. Turn the flag off, or compile with the default "
            "splitting."
        )


class AddAiterRMSNormPadPattern:
    """This pattern replaces an aiter_rmsnorm_with_add & a pad op
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
    """This pass replaces an AITER CK RMSNorm + residual add and a pad op
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
    """Fuse paired q_a_layernorm + kv_a_layernorm in MLA attention into
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


class MLADualRMSPerTokenQuantPattern(
    VllmPatternReplacement[
        ...,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]
):
    """Fuse the MLA FP8 attention path -- q-latent RMSNorm + FP8 *per-token* quant
    plus kv-latent RMSNorm -- into AITER's ``fused_qk_rmsnorm_per_token_quant``.

    With a per-token FP8 ``q_b_proj`` (Quark / ModelOpt), the earlier
    ``RocmAiterRMSNormQuantFusionPass`` folds the q side into
    ``rocm_aiter_rmsnorm_fused_dynamic_quant`` and leaves the kv side a plain
    ``vllm_ir.rms_norm``. This pattern matches that asymmetric pair::

        gemm -> split_with_sizes([q_dim, kv_dim])
            +-- q_c     -> rocm_aiter_rmsnorm_fused_dynamic_quant -> (q_fp8, q_scale)
            +-- kv_lora -> split_with_sizes([kv_c_dim, k_pe_dim])
                            +-- kv_c -> vllm_ir.rms_norm -> kv_normed (bf16)
                            +-- k_pe
    """

    DYNAMIC_QUANT_OP = rocm_aiter_ops.get_rmsnorm_fused_dynamic_quant_op()
    FUSED_OP = rocm_aiter_ops.get_fused_mla_dual_rms_norm_per_token_quant_op()

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def get_inputs(self) -> list[torch.Tensor]:
        q_dim, kv_c_dim, k_pe_dim = 256, 128, 64
        return [
            self.empty_bf16(5, q_dim + kv_c_dim + k_pe_dim),
            self.empty_bf16(q_dim),
            self.empty_bf16(kv_c_dim),
        ]

    @property
    def pattern(
        self,
    ) -> Callable[
        ...,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        eps = self._epsilon
        dynamic_quant_op = self.DYNAMIC_QUANT_OP

        def _pattern(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            q_dim = q_weight.shape[0]
            kv_dim = projected.shape[-1] - q_dim
            kv_c_dim = kv_weight.shape[0]
            k_pe_dim = kv_dim - kv_c_dim
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            q_quant = dynamic_quant_op(
                x=q_c,
                weight=q_weight,
                epsilon=eps,
                quant_dtype=FP8_DTYPE,
            )
            kv_normed = vllm.ir.ops.rms_norm(kv_c, kv_weight, eps)
            return q_quant[0], q_quant[1], kv_normed, k_pe

        return _pattern

    @property
    def replacement(
        self,
    ) -> Callable[
        ...,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        eps = self._epsilon
        fused_op = self.FUSED_OP

        def _replacement(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            q_dim = q_weight.shape[0]
            kv_dim = projected.shape[-1] - q_dim
            kv_c_dim = kv_weight.shape[0]
            k_pe_dim = kv_dim - kv_c_dim
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            at = fused_op(
                q_c,
                q_weight,
                kv_c,
                kv_weight,
                eps,
                eps,
            )
            # q_fp8, q_scale, kv_normed, k_pe
            return at[0], at[1], at[2], k_pe

        return _replacement


class MLADualRMSGroupQuantPattern(
    VllmPatternReplacement[
        ...,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]
):
    """Fuse the MLA FP8 *block-scale* attention path -- q-latent RMSNorm + FP8
    group quant plus kv-latent RMSNorm -- into AITER's
    ``fused_qk_rmsnorm_group_quant``.

    With a block-scale FP8 ``q_b_proj`` (DeepSeek fp8), the earlier
    ``RocmAiterRMSNormQuantFusionPass`` folds the q side into
    ``rocm_aiter_rmsnorm_fp8_group_quant`` (the Triton rms+group-quant kernel)
    and leaves the kv side a plain ``vllm_ir.rms_norm``. This pattern matches
    that asymmetric pair::

        gemm -> split_with_sizes([q_dim, kv_dim])
            +-- q_c     -> rocm_aiter_rmsnorm_fp8_group_quant -> (q_fp8, q_scale)
            +-- kv_lora -> split_with_sizes([kv_c_dim, k_pe_dim])
                            +-- kv_c -> vllm_ir.rms_norm -> kv_normed (bf16)
                            +-- k_pe
    """

    GROUP_QUANT_OP = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()
    FUSED_OP = rocm_aiter_ops.get_fused_mla_dual_rms_norm_group_quant_op()

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def get_inputs(self) -> list[torch.Tensor]:
        q_dim, kv_c_dim, k_pe_dim = 256, 128, 64
        return [
            self.empty_bf16(5, q_dim + kv_c_dim + k_pe_dim),
            self.empty_bf16(q_dim),
            self.empty_bf16(kv_c_dim),
        ]

    @property
    def pattern(
        self,
    ) -> Callable[
        ...,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        eps = self._epsilon
        group_quant_op = self.GROUP_QUANT_OP

        def _pattern(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            q_dim = q_weight.shape[0]
            kv_dim = projected.shape[-1] - q_dim
            kv_c_dim = kv_weight.shape[0]
            k_pe_dim = kv_dim - kv_c_dim
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            quant_kwargs = dict(
                x=q_c,
                weight=q_weight,
                variance_epsilon=eps,
                group_size=128,
            )
            q_quant = group_quant_op(**quant_kwargs)
            kv_normed = vllm.ir.ops.rms_norm(kv_c, kv_weight, eps)
            return q_quant[0], q_quant[1], kv_normed, k_pe

        return _pattern

    @property
    def replacement(
        self,
    ) -> Callable[
        ...,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        eps = self._epsilon
        fused_op = self.FUSED_OP

        def _replacement(
            projected: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            q_dim = q_weight.shape[0]
            kv_dim = projected.shape[-1] - q_dim
            kv_c_dim = kv_weight.shape[0]
            k_pe_dim = kv_dim - kv_c_dim
            q_c, kv_lora = projected.split([q_dim, kv_dim], dim=-1)
            kv_c, k_pe = kv_lora.split([kv_c_dim, k_pe_dim], dim=-1)
            at = fused_op(
                q_c,
                q_weight,
                kv_c,
                kv_weight,
                eps,
                eps,
                128,
                False,
            )
            # q_fp8, q_scale, kv_normed, k_pe
            return at[0], at[1], at[2], k_pe

        return _replacement


class MLADualRMSNormFusionPass(VllmFusionPatternMatcherPass):
    """Post-grad PatternMatcher pass that fuses paired q / kv RMS norms in
    MLA attention into ``fused_mla_dual_rms_norm`` backed by aiter's
    ``fused_qk_rmsnorm`` HIP kernel.

    The FP8 attention path is also handled via
    :class:`MLADualRMSPerTokenQuantPattern` or :class:`MLADualRMSGroupQuantPattern`,
    which fuse the q-latent RMSNorm + FP8 per-token/group quant together with
    the kv-latent RMSNorm into ``fused_mla_dual_rms_norm_<per_token|group>_quant``
    backed by aiter's ``fused_qk_rmsnorm_<per_token|group>_quant`` HIP kernel.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "mla_dual_rms_norm_fusion_pass")

        for epsilon in [1e-5, 1e-6]:
            self.register(MLADualRMSNormPattern(epsilon))
            self.register(MLADualRMSPerTokenQuantPattern(epsilon))
            self.register(MLADualRMSGroupQuantPattern(epsilon))
