# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE activation expressions for Humming input processing."""

from dataclasses import replace

from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
)

_SILU_IMPL = "a / (1.f + expf(-a))"
_GELU_IMPL = "0.5f * a * (1.f + erff(a * 0.7071067811865476f))"
_RELU2_IMPL = "fmaxf(a, 0.f) * fmaxf(a, 0.f)"
_GELU_TANH_IMPL = """[](float a) {
    const float cubic = 0.044715f * a * a * a;
    const float inner = 0.7978845608028654f * (a + cubic);
    return 0.5f * a * (1.f + tanhf(inner));
}(a)"""
_CLAMPED_SILU_IMPL = """[](float a, float b) {{
    const float gate = fminf(a, {limit});
    const float up = fmaxf(fminf(b, {limit}), -{limit});
    const float activated_gate = gate / (1.f + expf(-{alpha} * gate));
    return activated_gate * (up + {beta});
}}(a, b)"""

# Parameterized expressions are resolved by get_humming_activation().
_HUMMING_ACTIVATIONS: dict[MoEActivation, dict[str, str]] = {
    MoEActivation.SILU: {
        "activation_type": "binary_split",
        "activation_impl": f"({_SILU_IMPL}) * b",
    },
    MoEActivation.GELU: {
        "activation_type": "binary_split",
        "activation_impl": f"({_GELU_IMPL}) * b",
    },
    MoEActivation.GELU_TANH: {
        "activation_type": "binary_split",
        "activation_impl": f"({_GELU_TANH_IMPL}) * b",
    },
    MoEActivation.RELU2: {
        "activation_type": "binary_split",
        "activation_impl": f"({_RELU2_IMPL}) * b",
    },
    MoEActivation.SWIGLUOAI: {
        "activation_type": "binary_interleaved",
        "activation_impl": _CLAMPED_SILU_IMPL,
    },
    MoEActivation.SITU: {
        "activation_type": "binary_split",
        "activation_impl": """[](float a, float b) {{
            const float gate = {situ_beta} * tanhf(a / {situ_beta});
            const float activated_gate = gate / (1.f + expf(-a));
            const float up = {situ_up};
            return activated_gate * up;
        }}(a, b)""",
    },
    MoEActivation.SWIGLUOAI_UNINTERLEAVE: {
        "activation_type": "binary_split",
        "activation_impl": _CLAMPED_SILU_IMPL,
    },
    MoEActivation.SWIGLUSTEP: {
        "activation_type": "binary_split",
        "activation_impl": """[](float a, float b) {
            const float activated_gate = a / (1.f + expf(-a));
            const float gate = fminf(activated_gate, 7.f);
            const float up = fmaxf(fminf(b, 7.f), -7.f);
            return gate * up;
        }(a, b)""",
    },
    MoEActivation.SILU_NO_MUL: {
        "activation_type": "unary",
        "activation_impl": _SILU_IMPL,
    },
    MoEActivation.GELU_NO_MUL: {
        "activation_type": "unary",
        "activation_impl": _GELU_IMPL,
    },
    MoEActivation.GELU_TANH_NO_MUL: {
        "activation_type": "unary",
        "activation_impl": _GELU_TANH_IMPL,
    },
    MoEActivation.RELU2_NO_MUL: {
        "activation_type": "unary",
        "activation_impl": _RELU2_IMPL,
    },
}


def get_humming_activation(
    activation: MoEActivation,
    config: ApplyMoEActivationConfig | None = None,
) -> dict[str, str]:
    config = config or ApplyMoEActivationConfig()
    use_clamped_silu = False
    if activation == MoEActivation.SWIGLUOAI:
        config = ApplyMoEActivationConfig(clamp_limit=7.0, alpha=1.702, beta=1.0)
        use_clamped_silu = True
    elif activation == MoEActivation.SILU and config.clamp_limit is not None:
        config = replace(config, alpha=1.0, beta=0.0)
        use_clamped_silu = True
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        if config.clamp_limit is None:
            raise ValueError("SWIGLUOAI_UNINTERLEAVE requires clamp_limit")
        use_clamped_silu = True

    definition = _HUMMING_ACTIVATIONS[activation].copy()
    if activation == MoEActivation.SITU:
        if config.activation_situ_beta is None:
            raise ValueError("SITU requires activation_situ_beta")
        linear_beta = config.activation_situ_linear_beta
        situ_up = "b"
        if linear_beta is not None and linear_beta > 0:
            situ_up = f"({float(linear_beta)}f) * tanhf(b / ({float(linear_beta)}f))"
        situ_beta = f"({float(config.activation_situ_beta)}f)"
        activation_impl = definition["activation_impl"]
        activation_impl = activation_impl.format(situ_beta=situ_beta, situ_up=situ_up)
        definition["activation_impl"] = activation_impl
    elif use_clamped_silu:
        assert config.clamp_limit is not None
        limit = f"({float(config.clamp_limit)}f)"
        alpha = f"({float(config.alpha)}f)"
        beta = f"({float(config.beta)}f)"
        activation_impl = _CLAMPED_SILU_IMPL.format(limit=limit, alpha=alpha, beta=beta)
        definition["activation_impl"] = activation_impl
    return definition
