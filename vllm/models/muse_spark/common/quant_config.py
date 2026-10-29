# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization-config helpers shared by the MuseSpark1x vendor stacks."""

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)

# llm-compressor writes regex targets against the HF module tree. Literal
# targets are remapped through the model's WeightsMapper, but ``re:`` patterns
# pass through untouched, so the HF names they embed are rewritten here.
_HF_TO_VLLM_REGEX_REWRITES: tuple[tuple[str, str], ...] = (
    (r"mlp\.(pre|post)_expert_proj", r"feed_forward\.(pooler|unpooler)"),
    (r"mlp\.gate$", r"feed_forward\.router$"),
    (r"mlp\.", r"feed_forward\."),
)


def _rewrite_regex_target(target: str) -> str:
    if not target.startswith("re:"):
        return target
    for hf_name, vllm_name in _HF_TO_VLLM_REGEX_REWRITES:
        target = target.replace(hf_name, vllm_name)
    return target


def align_compressed_tensors_quant_config(
    quant_config: CompressedTensorsConfig,
) -> None:
    """Point the config's regex targets and ignores at vLLM module names.

    Without this, ``re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$`` never
    matches ``feed_forward.experts`` and the experts silently build as BF16,
    then fail to load the packed MXFP4 tensors. Idempotent.
    """
    quant_config.target_scheme_map = {
        _rewrite_regex_target(target): scheme
        for target, scheme in quant_config.target_scheme_map.items()
    }
    quant_config.ignore = [
        _rewrite_regex_target(target) for target in quant_config.ignore
    ]


__all__ = ["align_compressed_tensors_quant_config"]
