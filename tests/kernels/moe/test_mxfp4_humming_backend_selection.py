# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection-logic tests for `--moe-backend flashinfer_cutlass_humming`.

The backend maps to the FlashInfer CUTLASS SM90 MXFP4-weight x FP8-activation
"humming" kernel. These tests are CPU-only: they cover argument plumbing,
the string -> enum mapping, and the guards that keep a misdirected request
from reaching the kernel, not the kernel itself.
"""

from typing import get_args
from unittest.mock import patch

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import get_attr_docs
from vllm.config.kernel import KernelConfig, MoEBackend
from vllm.engine.arg_utils import EngineArgs, get_kwargs
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _check_explicit_backend_platform,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    map_mxfp4_backend,
    select_deepseek_v4_mxfp4_moe_backend,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

HUMMING_BACKEND = "flashinfer_cutlass_humming"

ORACLE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"


def _parse(argv: list[str]) -> EngineArgs:
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(parser.parse_args(argv))


def test_literal_contains_backend():
    assert HUMMING_BACKEND in get_args(MoEBackend)


def test_kernel_config_accepts_backend():
    assert KernelConfig(moe_backend=HUMMING_BACKEND).moe_backend == HUMMING_BACKEND


def test_engine_args_accepts_backend():
    assert EngineArgs(moe_backend=HUMMING_BACKEND).moe_backend == HUMMING_BACKEND


def test_cli_normalizes_dashed_form():
    """`--moe-backend` lowercases and turns dashes into underscores."""
    args = _parse(["--moe-backend", "flashinfer-cutlass-humming"])
    assert args.moe_backend == HUMMING_BACKEND


def test_cli_rejects_unknown_backend():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(["--moe-backend", "flashinfer_cutlass_hum"])


def test_cli_offers_backend_as_a_choice():
    assert HUMMING_BACKEND in get_kwargs(KernelConfig)["moe_backend"]["choices"]


def test_cli_help_disambiguates_the_two_hummings():
    """The `--help` text must separate this backend from the `humming` package
    backend, since the names collide."""
    help_text = get_attr_docs(KernelConfig)["moe_backend"]
    assert HUMMING_BACKEND in help_text
    assert "third-party" in help_text


def test_map_backend_string():
    assert map_mxfp4_backend(HUMMING_BACKEND) == [
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_FP8_HUMMING
    ]


def test_map_backend_does_not_steal_humming_package_backend():
    """`--moe-backend humming` must keep pointing at the third-party package."""
    assert map_mxfp4_backend("humming") == [Mxfp4MoeBackend.HUMMING]


def test_explicit_request_on_non_sm90_is_actionable():
    config = make_dummy_moe_config()
    config.moe_backend = HUMMING_BACKEND
    with patch(f"{ORACLE}.current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.is_device_capability.return_value = False
        platform.get_device_capability.return_value = 120
        with pytest.raises(ValueError, match="SM90"):
            select_deepseek_v4_mxfp4_moe_backend(config)


def test_explicit_request_on_sm90_passes_the_guard():
    with patch(f"{ORACLE}.current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.is_device_capability.side_effect = lambda c: c == 90
        _check_explicit_backend_platform(
            HUMMING_BACKEND, map_mxfp4_backend(HUMMING_BACKEND)
        )


@pytest.mark.parametrize("backend", ["marlin", "humming", "triton_unfused"])
def test_guard_only_fires_for_the_flashinfer_humming_backend(backend):
    with patch(f"{ORACLE}.current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.is_device_capability.return_value = False
        _check_explicit_backend_platform(backend, map_mxfp4_backend(backend))


def _dummy_mxfp4_weights(num_experts=2, intermediate_size=128, hidden_size=128):
    """MXFP4 payloads as stored on the layer: w13 is (E, 2I, K // 2) uint8."""
    w13 = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size // 2), dtype=torch.uint8
    )
    w2 = torch.zeros(
        (num_experts, hidden_size, intermediate_size // 2), dtype=torch.uint8
    )
    w13_scale = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size // 32), dtype=torch.uint8
    )
    w2_scale = torch.zeros(
        (num_experts, hidden_size, intermediate_size // 32), dtype=torch.uint8
    )
    return w13, w2, w13_scale, w2_scale


def test_gpt_oss_layout_is_refused():
    """GPT-OSS stores w13 row-interleaved; the humming weight preprocessing
    only handles the DeepSeek-V4 block layout, so it must refuse rather than
    return silently wrong numerics."""
    w13, w2, w13_scale, w2_scale = _dummy_mxfp4_weights()
    with pytest.raises(ValueError, match="DeepSeek-V4 MXFP4 expert layout"):
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_FP8_HUMMING,
            torch.nn.Module(),
            w13,
            w2,
            w13_scale,
            w2_scale,
        )


@pytest.mark.parametrize(
    "intermediate_size,hidden_size", [(64, 128), (128, 64), (192, 128)]
)
def test_unaligned_shapes_are_refused(intermediate_size, hidden_size):
    """Both GEMM dims must be multiples of 128; report which one is not
    instead of failing inside the FlashInfer interleave."""
    w13, w2, w13_scale, w2_scale = _dummy_mxfp4_weights(
        intermediate_size=intermediate_size, hidden_size=hidden_size
    )
    with pytest.raises(ValueError, match="multiples of 128"):
        convert_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_FP8_HUMMING,
            torch.nn.Module(),
            w13,
            w2,
            w13_scale,
            w2_scale,
        )
