# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection-logic tests for `--moe-backend flashinfer_cutlass_humming`.

The backend maps to the FlashInfer CUTLASS SM90 MXFP4-weight x FP8-activation
"humming" kernel. These tests are CPU-only: they cover argument plumbing,
the string -> enum mapping, and the guards that keep a misdirected request
from reaching the kernel, not the kernel itself.
"""

from contextlib import contextmanager
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
    _check_explicit_backend_requirements,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    convert_weight_to_mxfp4_moe_kernel_format,
    map_mxfp4_backend,
    select_deepseek_v4_mxfp4_moe_backend,
)

HUMMING_BACKEND = "flashinfer_cutlass_humming"

ORACLE = "vllm.model_executor.layers.fused_moe.oracle.mxfp4"


# A new `--moe-backend` value only becomes usable once it has landed in all of
# these; each is a separate mechanism and any one of them can be missed without
# the others noticing.
_CONFIG_SURFACES = {
    "literal": lambda: HUMMING_BACKEND in get_args(MoEBackend),
    "kernel_config": lambda: KernelConfig(moe_backend=HUMMING_BACKEND).moe_backend
    == HUMMING_BACKEND,
    "engine_args": lambda: EngineArgs(moe_backend=HUMMING_BACKEND).moe_backend
    == HUMMING_BACKEND,
    "cli_choices": lambda: HUMMING_BACKEND
    in get_kwargs(KernelConfig)["moe_backend"]["choices"],
}


@pytest.mark.parametrize("surface", _CONFIG_SURFACES)
def test_backend_name_reaches_every_config_surface(surface):
    assert _CONFIG_SURFACES[surface]()


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


@contextmanager
def _patched_platform(*, is_sm90: bool, humming_available: bool = True):
    """Run the oracle against a synthetic device and FlashInfer build."""
    with (
        patch(f"{ORACLE}.current_platform") as platform,
        patch(f"{ORACLE}.has_flashinfer_humming_moe", return_value=humming_available),
    ):
        platform.is_cuda.return_value = True
        platform.is_rocm.return_value = False
        platform.is_device_capability.side_effect = lambda c: is_sm90 and c == 90
        platform.has_device_capability.side_effect = (
            lambda c: (90 if is_sm90 else 120) >= c
        )
        platform.get_device_capability.return_value = 90 if is_sm90 else 120
        yield platform


def test_explicit_request_on_non_sm90_is_actionable():
    config = make_dummy_moe_config()
    config.moe_backend = HUMMING_BACKEND
    with _patched_platform(is_sm90=False), pytest.raises(ValueError, match="SM90"):
        select_deepseek_v4_mxfp4_moe_backend(config)


def test_explicit_request_on_sm90_passes_the_guard():
    with _patched_platform(is_sm90=True):
        _check_explicit_backend_requirements(
            HUMMING_BACKEND, map_mxfp4_backend(HUMMING_BACKEND)
        )


def test_explicit_request_on_old_flashinfer_names_the_version():
    """Without the probe this surfaces as an ImportError from deep inside
    weight loading; the user needs to be told which version to install, and the
    version named has to be the one the probe actually accepts."""
    config = make_dummy_moe_config()
    config.moe_backend = HUMMING_BACKEND
    with (
        _patched_platform(is_sm90=True, humming_available=False),
        pytest.raises(ValueError, match="flashinfer-python>=0.6.18"),
    ):
        select_deepseek_v4_mxfp4_moe_backend(config)


def test_auto_selection_never_reaches_the_backend():
    """The backend is opt-in only: it appears in no automatic priority list, so
    a DeepSeek-V4 MXFP4 model on SM90 must not land on it without an explicit
    request, even where the kernel is available."""
    config = make_dummy_moe_config()  # moe_backend stays "auto"
    with _patched_platform(is_sm90=True):
        try:
            backend, _ = select_deepseek_v4_mxfp4_moe_backend(config)
        except (ValueError, NotImplementedError):
            pass  # nothing suitable on this synthetic device; still not humming
        else:
            assert backend is not Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_FP8_HUMMING


@pytest.mark.parametrize("backend", ["marlin", "humming", "triton_unfused"])
def test_guard_only_fires_for_the_flashinfer_humming_backend(backend):
    with _patched_platform(is_sm90=False, humming_available=False):
        _check_explicit_backend_requirements(backend, map_mxfp4_backend(backend))


def test_probe_ignores_symbols_the_humming_path_never_uses():
    """The probe must not require nvfp4/TRT-LLM symbols. FlashInfer builds
    exist that carry the humming kernel but not `fp4_quantize` /
    `nvfp4_block_scale_interleave`; rejecting those would tell the user to
    upgrade a FlashInfer that already works."""
    from types import SimpleNamespace

    from vllm.utils import flashinfer as fi

    def cutlass_fused_moe(*, use_wfp4afp8_humming=False): ...

    humming_only = SimpleNamespace(
        cutlass_fused_moe=cutlass_fused_moe,
        preprocess_moe_weights_for_sm90_mixed_gemm_humming=lambda *a, **k: None,
        interleave_moe_weights_for_sm90_mixed_gemm=lambda *a, **k: None,
        interleave_moe_scales_for_sm90_mixed_gemm=lambda *a, **k: None,
    )
    fi.has_flashinfer_humming_moe.cache_clear()
    with (
        patch.object(fi, "has_flashinfer_moe", return_value=True),
        patch.object(fi, "_get_submodule", return_value=humming_only),
        patch.object(fi, "_has_per_local_expert_residual", return_value=True),
    ):
        assert fi.has_flashinfer_humming_moe() is True
    fi.has_flashinfer_humming_moe.cache_clear()


def test_probe_rejects_a_build_that_predates_the_per_local_expert_residual():
    """FlashInfer #3738 shipped the humming kernel with per-routed-token
    residuals and #4431 changed them to per-local-expert. Every symbol and
    keyword this probe used to look at is present in both, so a #3738-only
    build would be selected and then handed the wrong contract -- silently
    wrong whenever `num_tokens * top_k` equals the local expert count."""
    from types import SimpleNamespace

    from vllm.utils import flashinfer as fi

    def cutlass_fused_moe(*, use_wfp4afp8_humming=False): ...

    pre_4431 = SimpleNamespace(
        cutlass_fused_moe=cutlass_fused_moe,
        preprocess_moe_weights_for_sm90_mixed_gemm_humming=lambda *a, **k: None,
        interleave_moe_weights_for_sm90_mixed_gemm=lambda *a, **k: None,
        interleave_moe_scales_for_sm90_mixed_gemm=lambda *a, **k: None,
    )
    fi.has_flashinfer_humming_moe.cache_clear()
    with (
        patch.object(fi, "has_flashinfer_moe", return_value=True),
        patch.object(fi, "_get_submodule", return_value=pre_4431),
        patch.object(fi, "_has_per_local_expert_residual", return_value=False),
    ):
        assert fi.has_flashinfer_humming_moe() is False
    fi.has_flashinfer_humming_moe.cache_clear()


@pytest.mark.parametrize(
    "shape_check,expected",
    [
        ("fc1 residual scale must have one element per local expert", True),
        ("fc1 token scale must have one element per routed token", False),
    ],
)
def test_residual_contract_is_read_off_the_shipped_jit_source(shape_check, expected):
    """The two builds differ only in C++, so the probe reads the kernel-side
    shape check that FlashInfer ships as JIT source next to the package."""
    from vllm.utils import flashinfer as fi

    class _FakeSource:
        def __truediv__(self, _part):
            return self

        def read_bytes(self):
            return shape_check.encode()

    fi._has_per_local_expert_residual.cache_clear()
    with patch.object(fi.importlib.resources, "files", return_value=_FakeSource()):
        assert fi._has_per_local_expert_residual() is expected
    fi._has_per_local_expert_residual.cache_clear()


def test_residual_contract_probe_survives_a_missing_source_tree():
    """Some redistributions strip the JIT sources. Failing closed costs the
    user this backend; failing open would hand the kernel a wrong contract."""
    from vllm.utils import flashinfer as fi

    fi._has_per_local_expert_residual.cache_clear()
    with patch.object(fi.importlib.resources, "files", side_effect=FileNotFoundError):
        assert fi._has_per_local_expert_residual() is False
    fi._has_per_local_expert_residual.cache_clear()


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
