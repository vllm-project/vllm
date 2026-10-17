# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe AR/Gemma RMSNorm patterns, including static FP8 output."""

import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch import fx
from torch._higher_order_ops import auto_functionalized
from torch._inductor import pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir
from vllm.compilation.passes.inductor_pass import enable_fake_mode
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators import cute_allreduce as runtime


def _bind_optional_inputs(fn: Callable[..., Any], residual: bool, quantized: bool):
    if residual and quantized:
        return lambda input, residual, weight, scale: fn(input, residual, weight, scale)
    if residual:
        return lambda input, residual, weight: fn(input, residual, weight, None)
    if quantized:
        return lambda input, weight, scale: fn(input, None, weight, scale)
    return lambda input, weight: fn(input, None, weight, None)


class CuteAllReduceFusionPass(VllmPatternMatcherPass):
    def __init__(self, config):
        super().__init__(config)
        self.disabled = not runtime.enabled_for_config(config)
        self.patterns = PatternMatcherPass(pass_name="cute_allreduce")
        self.policy_key = "unavailable"
        if self.disabled:
            return
        backend = runtime.get_backend()
        assert backend is not None
        self.policy_key = repr(backend.policy)
        self.hidden_size = config.model_config.get_hidden_size()
        epsilon = float(config.model_config.hf_text_config.rms_norm_eps)
        self.register_patterns(epsilon)

    @enable_fake_mode
    def register_patterns(self, epsilon):
        # Prefer FP8 when the BF16 norm has no other consumer; otherwise the
        # BF16 pattern keeps all consumers and any following quantizers intact.
        for quantized in (True, False):
            for add_residual in (True, False):
                for keep_residual in (True, False):
                    group_shapes = (None, [-1, -1]) if quantized else (None,)
                    for group_shape in group_shapes:
                        self._register(
                            epsilon, quantized, add_residual, keep_residual, group_shape
                        )

    def _register(self, epsilon, quantized, add_residual, keep_residual, group_shape):
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8StaticTensorSym,
        )

        from .matcher_utils import MatcherQuantFP8

        quantizer = MatcherQuantFP8(kFp8StaticTensorSym) if quantized else None
        op = torch.ops.vllm.cute_allreduce_norm.default

        def norm(input, residual, weight, scale):
            reduced = tensor_model_parallel_all_reduce(input)
            if add_residual:
                normalized, updated = vllm.ir.ops.fused_add_rms_norm(
                    reduced, residual, weight.float() + 1.0, epsilon
                )
            else:
                normalized = vllm.ir.ops.rms_norm(
                    reduced, weight.float() + 1.0, epsilon
                )
                updated = reduced
            if quantized:
                assert quantizer is not None
                if quantizer.enabled:
                    # Current quantizers emit the optional group_shape argument
                    # explicitly. Match both per-tensor spellings in the schema.
                    result = torch.empty(
                        normalized.shape,
                        device=normalized.device,
                        dtype=torch.float8_e4m3fn,
                    )
                    _, normalized = auto_functionalized(
                        quantizer.QUANT_OP,
                        result=result,
                        input=normalized,
                        scale=scale,
                        group_shape=group_shape,
                    )
                else:
                    normalized, _ = quantizer(normalized, scale)
            return (normalized, updated) if keep_residual else normalized

        def replace(input, residual, weight, scale):
            value = op(input, residual, weight, scale, epsilon)
            return value if keep_residual else value[0]

        x = torch.empty((5, 16), dtype=torch.bfloat16, device="cuda")
        w = torch.empty(16, dtype=torch.bfloat16, device="cuda")
        scale = torch.empty(1, dtype=torch.float32, device="cuda")
        inputs = [x]
        if add_residual:
            inputs.append(torch.empty_like(x))
        inputs.append(w)
        if quantized:
            inputs.append(scale)
        pattern = _bind_optional_inputs(norm, add_residual, quantized)
        replacement = _bind_optional_inputs(replace, add_residual, quantized)

        def eligible(match):
            value = match.kwargs["input"].meta.get("val")
            weight = match.kwargs["weight"].meta.get("val")
            if not (
                isinstance(value, torch.Tensor)
                and value.ndim == 2
                and value.shape[-1] == self.hidden_size
                and value.dtype == torch.bfloat16
                and value.is_contiguous()
                and isinstance(weight, torch.Tensor)
                and weight.dtype == torch.bfloat16
                and weight.shape == (self.hidden_size,)
                and weight.is_contiguous()
            ):
                return False
            if add_residual:
                residual = match.kwargs["residual"].meta.get("val")
                if not (
                    isinstance(residual, torch.Tensor)
                    and residual.dtype == value.dtype
                    and residual.shape == value.shape
                    and residual.is_contiguous()
                ):
                    return False
            if quantized:
                qscale = match.kwargs["scale"].meta.get("val")
                return (
                    isinstance(qscale, torch.Tensor)
                    and qscale.dtype == torch.float32
                    and qscale.numel() == 1
                    and qscale.is_contiguous()
                    and qscale.device == value.device
                )
            return True

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            self.patterns,
            extra_check=eligible,
        )
        pm._seen_patterns.clear()

    def is_applicable_for_range(self, compile_range):
        return not self.disabled and compile_range.end <= runtime.MAX_TOKENS

    def uuid(self):
        return self.hash_source(
            type(self),
            _bind_optional_inputs,
            runtime.cute_allreduce_norm,
            inspect.unwrap(runtime.build_policy),
            self.policy_key,
            str(self.disabled),
        )

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            return
        self.matched_count = self.patterns.apply(graph)
        self.match_table[self.pass_name] += self.matched_count
        if self.matched_count:
            graph.eliminate_dead_code()
            graph.lint()
