# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformers modeling backend utilities."""

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from vllm.config.utils import getattr_iter
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig


logger = init_logger(__name__)


# Copied from `accelerate`
@contextmanager
def init_on_device_without_buffers(device: torch.device):
    """
    A context manager under which models are initialized with all
    parameters on the specified device. However buffers are not
    initialized on specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
    """

    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        for torch_function_name in tensor_constructors_to_patch:
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


Style = Literal["colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"]


def replace_linear_class(
    linear: nn.Linear,
    style: Style = "replicate",
    quant_config: "QuantizationConfig | None" = None,
    *,
    prefix: str = "",
) -> ColumnParallelLinear | RowParallelLinear | ReplicatedLinear:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

    Args:
        linear: `nn.Linear` to be replaced.
        style: Tensor parallel style of the new linear, e.g. "colwise".
        quant_config: Quantization config for the new linear.
    Returns:
        The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")

    vllm_linear_cls, vllm_linear_kwargs = {
        "colwise": (ColumnParallelLinear, {}),
        "colwise_rep": (ColumnParallelLinear, {"gather_output": True}),
        "rowwise": (RowParallelLinear, {}),
        "rowwise_rep": (RowParallelLinear, {"input_is_parallel": False}),
        "replicate": (ReplicatedLinear, {}),
    }.get(style, (ReplicatedLinear, {}))

    return vllm_linear_cls(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
        prefix=prefix,
        return_bias=False,
        **vllm_linear_kwargs,
    )


def replace_rms_norm_class(rms_norm: nn.Module, hidden_size: int) -> RMSNorm:
    """Replace a Transformers RMSNorm with vLLM's RMSNorm.

    This method assumes:
    - Weight is stored as `weight`.
    - Epsilon is stored as `eps` or `variance_epsilon`.
    - `with_scale` indicates whether the layer has a weight (Gemma3n only).
    - `var_hidden_size` is only ever used for Intern vision encoder in vLLM
    and Transformers doesn't appear to have the same concept.
    """
    eps = getattr_iter(rms_norm, ("eps", "variance_epsilon"), 1e-6)
    kwargs = {"hidden_size": hidden_size, "eps": eps}
    # Update hidden size if weight is available
    weight_meta = getattr(rms_norm, "weight", None)
    if weight_meta is not None:
        kwargs["hidden_size"] = weight_meta.size(0)
    # Check if weight is all zeros, which indicates GemmaRMSNorm
    # We must create a new instance because rms_norm is on meta
    try:
        with torch.device("cpu"):
            weight_test = getattr(rms_norm.__class__(1), "weight", None)
    except Exception:
        logger.warning(
            "Failed to determine if RMSNorm weight is centered on zero or one. "
            "Defaulting to one."
        )
        weight_test = None
    if weight_test is not None and torch.all(weight_test == 0):
        return GemmaRMSNorm(**kwargs)
    # Otherwise assume it's a regular RMSNorm
    kwargs["has_weight"] = getattr(rms_norm, "with_scale", True)
    if weight_meta is not None:
        kwargs["dtype"] = weight_meta.dtype
    else:
        # No weight, fall back to weightless RMSNorm
        kwargs["has_weight"] = False
    return RMSNorm(**kwargs)


def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module):
    logger.debug("%s: %s -> %s", name, old_module, new_module)


def get_feature_request_tip(
    model: str,
    trust_remote_code: bool,
) -> str:
    hf_url = f"a discussion at https://huggingface.co/{model}/discussions/new"
    gh_url = "an issue at https://github.com/huggingface/transformers/issues/new/choose"
    url = hf_url if trust_remote_code else gh_url
    prefix = f"Please open {url} to request support for this feature. "
    if Path(model).exists():
        prefix = ""
    doc_url = "https://docs.vllm.ai/en/latest/models/supported_models.html#writing-custom-models"
    tip = f"See {doc_url} for instructions on how to add support yourself."
    return f"{prefix}{tip}"


def can_enable_torch_compile(vllm_config: "VllmConfig") -> bool:
    """
    Callable to be passed to `@support_torch_compile`'s `enable_if` argument.

    Defaults to `True` but is disabled in the following situations:

    - The model uses dynamic rope scaling.
    """
    text_config = vllm_config.model_config.hf_config.get_text_config()
    # Dynamic rope scaling is not compatible with torch.compile
    rope_scaling: dict = getattr(text_config, "rope_scaling", None) or {}
    return rope_scaling.get("rope_type") != "dynamic"
