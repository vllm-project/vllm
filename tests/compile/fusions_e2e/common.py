# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import pytest
import regex as re

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class Matches(NamedTuple):
    # simple pointwise
    rms_quant_fusion: int = 0
    act_quant_fusion: int = 0
    norm_rope_fusion: int = 0
    attn_quant_fusion: int = 0
    # distributed
    ar_rms_fusion: int = 0
    sequence_parallel: int = 0
    async_tp: int = 0


class ModelFusionInfo(NamedTuple):
    model_name: str
    matches: Callable[[int], Matches]
    """Given number of hidden layers, produces the matches object"""
    # default_num_layers: int
    # """The default number of layers for this model"""
    model_kwargs: dict[str, Any] = {}
    hf_overrides: Callable[[int], dict] = lambda n: {"num_hidden_layers": n}


class AttentionBackendCase(NamedTuple):
    backend: AttentionBackendEnum
    model_kwargs: dict[str, Any] = {}
    """Additional args required for attn+quant fusion"""


is_blackwell = lambda: current_platform.is_device_capability_family(100)
"""Are we running on Blackwell, a lot of tests depend on it"""


def custom_ops_combos(*custom_ops: str) -> Iterable[str]:
    """Generate all combinations of custom ops for parametrization."""
    custom_ops_lists = [[f"-{op}", f"+{op}"] for op in custom_ops]
    for op_list in itertools.product(*custom_ops_lists):
        yield ",".join(op_list)


# Quick inline validation
assert list(custom_ops_combos("silu_and_mul")) == ["-silu_and_mul", "+silu_and_mul"]
assert list(custom_ops_combos("quant_fp8", "rms_norm")) == [
    "-quant_fp8,-rms_norm",
    "-quant_fp8,+rms_norm",
    "+quant_fp8,-rms_norm",
    "+quant_fp8,+rms_norm",
]


def has_cuda_graph_wrapper_metadata() -> bool:
    from importlib import import_module

    try:
        module = import_module("torch._inductor.utils")
        module.CUDAGraphWrapperMetadata  # noqa B018
    except AttributeError:
        return False
    return True


INDUCTOR_GRAPH_PARTITION = [
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not has_cuda_graph_wrapper_metadata(),
            reason="torch version does not support Inductor partition",
        ),
        id="inductor_partition",
    ),
    pytest.param(False, id="dynamo_partition"),
]

FUSION_LOG_PATTERNS: dict[str, re.Pattern] = {
    "rms_quant_fusion": re.compile(
        r"\[(?:compilation/)?fusion.py:\d+] Replaced (\d+) patterns"
    ),
    "act_quant_fusion": re.compile(
        r"activation_quant_fusion.py:\d+] Replaced (\d+) patterns"
    ),
    "norm_rope_fusion": re.compile(
        r"qk_norm_rope_fusion.py:\d+] Fused QK Norm\+RoPE on (\d+) sites"
    ),
    "attn_quant_fusion": re.compile(
        r"fusion_attn.py:\d+] Fused quant onto (\d+) attention nodes"
    ),
    "ar_rms_fusion": re.compile(r"collective_fusion.py:\d+] Replaced (\d+) patterns"),
    "sequence_parallel": re.compile(
        r"sequence_parallelism.py:\d+] Replaced (\d+) patterns"
    ),
    "async_tp": re.compile(r"collective_fusion.py:\d+] Replaced (\d+) patterns"),
}
