# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for reporting resolved quantization kernels in model inspection."""

import ast
import enum
import pathlib
from types import SimpleNamespace

import pytest
import regex as re
import torch
import torch.nn as nn

import vllm
import vllm.envs as envs
from vllm.config import DeviceConfig, LoadConfig, VllmConfig
from vllm.model_executor.kernels.linear import MarlinMxFp4LinearKernel
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader,
    log_model_inspection,
)
from vllm.model_inspection import (
    _BACKEND_ATTRIBUTES,
    _KERNEL_ATTRIBUTES,
    _get_quant_info,
    _get_selected_kernel,
    format_model_inspection,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _Experts:
    """Stand-in for a `FusedMoEExperts` implementation."""


class _MoEKernel:
    """Stand-in for `FusedMoEKernel`, which exposes its experts implementation."""

    def __init__(self) -> None:
        self.fused_experts = _Experts()


class _Backend(enum.Enum):
    MARLIN = "marlin"
    NONE = "none"


class _QuantMethod:
    """A quant method that holds an arbitrary set of resolution attributes."""

    def __init__(self, **attributes: object) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class _Linear(nn.Module):
    def __init__(self, quant_method: object) -> None:
        super().__init__()
        self.quant_method = quant_method

    def extra_repr(self) -> str:
        return "in_features=16, out_features=8"


class _CompressedTensorsLinear(nn.Module):
    """Compressed-tensors keeps both the scheme name and its kernel."""

    class _Scheme:
        def __init__(self) -> None:
            self.kernel = MarlinMxFp4LinearKernel

    def __init__(self) -> None:
        super().__init__()
        self.quant_method = _QuantMethod()
        self.scheme = self._Scheme()


class _UnquantizedLinear(nn.Module):
    class UnquantizedLinearMethod:
        pass

    def __init__(self) -> None:
        super().__init__()
        self.quant_method = self.UnquantizedLinearMethod()


@pytest.mark.parametrize(
    "attribute",
    [
        "moe_kernel",
        "fp8_linear",
        "linear_kernel",
        "ocp_mx_linear",
        "kernel",
        "experts_cls",
    ],
)
def test_get_selected_kernel_reports_every_known_attribute(attribute):
    """Each kernel attribute name used across quant methods is discovered."""
    method = _QuantMethod(**{attribute: MarlinMxFp4LinearKernel})
    assert _get_selected_kernel(method, None) == "MarlinMxFp4LinearKernel"


@pytest.mark.parametrize(
    "attribute",
    [
        "mxfp4_backend",
        "mxfp8_backend",
        "nvfp4_backend",
        "int8_backend",
        "fp8_backend",
        "w4a8_backend",
        "wna16_backend",
        "wna16_moe_backend",
        "unquantized_backend",
        "backend",
    ],
)
def test_get_selected_kernel_reports_every_backend_enum(attribute):
    """Every enum backend attribute name used across MoE methods is discovered."""
    method = _QuantMethod(**{attribute: _Backend.MARLIN})
    assert _get_selected_kernel(method, None) == "MARLIN"


def test_get_selected_kernel_prefers_a_kernel_over_a_stale_backend_enum():
    """The backend enum can be recorded before the final kernel is chosen.

    The MXFP4 MoE method defaults `mxfp4_backend` to MARLIN and then selects
    Cutlass experts without updating it, so the enum must not win.
    """
    method = _QuantMethod(
        mxfp4_backend=_Backend.MARLIN, experts_cls=MarlinMxFp4LinearKernel
    )
    assert _get_selected_kernel(method, None) == "MarlinMxFp4LinearKernel"


def test_get_selected_kernel_ignores_string_valued_backend_fields():
    """Some config objects expose a plain-string `backend` that is not a kernel."""
    method = _QuantMethod(backend="xpu")
    assert _get_selected_kernel(method, None) is None


def test_get_selected_kernel_names_the_experts_of_a_moe_kernel():
    """`FusedMoEKernel` has no name of its own, so its experts are reported."""
    method = _QuantMethod(moe_kernel=_MoEKernel())
    assert _get_selected_kernel(method, None) == "_MoEKernel(experts=_Experts)"


def test_get_selected_kernel_prefers_scheme_over_quant_method():
    """Compressed-tensors stores the kernel on the scheme."""
    scheme = _CompressedTensorsLinear._Scheme()
    method = _QuantMethod(kernel="WrongKernel")
    assert _get_selected_kernel(method, scheme) == "MarlinMxFp4LinearKernel"


def test_get_selected_kernel_is_none_before_selection():
    """Kernels are selected during weight processing, so absence is expected."""
    assert _get_selected_kernel(_QuantMethod(), None) is None


def test_get_quant_info_reports_the_kernel():
    module = _Linear(_QuantMethod(kernel=MarlinMxFp4LinearKernel))
    assert (
        _get_quant_info(module) == "quant=_QuantMethod(kernel=MarlinMxFp4LinearKernel)"
    )


def test_get_quant_info_reports_the_scheme_and_its_kernel():
    """Regression test: the compressed-tensors scheme holds the kernel."""
    assert (
        _get_quant_info(_CompressedTensorsLinear())
        == "quant=_Scheme(kernel=MarlinMxFp4LinearKernel)"
    )


def test_get_quant_info_omits_the_kernel_before_selection():
    assert _get_quant_info(_Linear(_QuantMethod())) == "quant=_QuantMethod"


def test_get_quant_info_skips_unquantized_layers_and_plain_modules():
    assert _get_quant_info(_UnquantizedLinear()) is None
    assert _get_quant_info(nn.Linear(4, 4)) is None


def test_format_model_inspection_reports_kernels_in_the_tree():
    """End-to-end check of the string that is logged for a model."""

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quantized = _Linear(_QuantMethod(kernel=MarlinMxFp4LinearKernel))
            self.unquantized = _UnquantizedLinear()

    output = format_model_inspection(Model())

    expected = "(quantized): _Linear(quant=_QuantMethod(kernel=MarlinMxFp4LinearKernel)"
    assert expected in output
    assert "(unquantized): _UnquantizedLinear()" in output


def test_format_model_inspection_separates_layers_with_different_kernels():
    """Layers that resolve to different kernels are not collapsed together."""

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    _Linear(_QuantMethod(kernel=MarlinMxFp4LinearKernel)),
                    _Linear(_QuantMethod(kernel=_Experts)),
                ]
            )

    output = format_model_inspection(Model())

    assert "MarlinMxFp4LinearKernel" in output
    assert "kernel=_Experts" in output


def test_log_model_inspection_is_disabled_by_default(monkeypatch):
    """The load-time log stays silent unless explicitly enabled."""
    monkeypatch.setattr(envs, "VLLM_LOG_MODEL_INSPECTION", False)
    logged = []
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.logger.info",
        lambda *args, **kwargs: logged.append(args),
    )

    log_model_inspection(nn.Linear(4, 4))

    assert logged == []


def test_log_model_inspection_logs_the_model_structure(monkeypatch):
    """When enabled, the log includes the resolved kernel."""
    monkeypatch.setattr(envs, "VLLM_LOG_MODEL_INSPECTION", True)
    logged = []
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.logger.info",
        lambda *args, **kwargs: logged.append(args),
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quantized = _Linear(_QuantMethod(kernel=MarlinMxFp4LinearKernel))

    log_model_inspection(Model())

    assert len(logged) == 1
    assert "MarlinMxFp4LinearKernel" in logged[0][1]


class _StubLoader(BaseModelLoader):
    """Loader that returns a model holding one unresolved quantized layer."""

    def __init__(self) -> None:
        super().__init__(LoadConfig())
        self.model = _Linear(_QuantMethod())

    def download_model(self, model_config) -> None:
        pass

    def load_weights(self, model, model_config) -> None:
        pass


def test_load_model_inspects_after_the_kernel_is_selected(monkeypatch):
    """The inspection log must observe the kernels selected during loading.

    Regression test for calling `log_model_inspection` from `create_model`,
    which runs before `process_weights_after_loading` and therefore reports no
    kernel for any MoE layer.
    """
    monkeypatch.setattr(envs, "VLLM_LOG_MODEL_INSPECTION", True)
    loader = _StubLoader()
    observed: list[str | None] = []

    def select_kernel(model, model_config, device) -> None:
        model.quant_method.kernel = MarlinMxFp4LinearKernel

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.process_weights_after_loading",
        select_kernel,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.initialize_model",
        lambda **kwargs: loader.model,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.log_online_quantization",
        lambda vllm_config: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.log_online_quantization_time",
        lambda vllm_config: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.log_model_inspection",
        lambda model: observed.append(_get_selected_kernel(model.quant_method, None)),
    )

    # `load_model` only reads `dtype` from the model config.
    model_config = SimpleNamespace(dtype=torch.float32)
    loader.load_model(VllmConfig(device_config=DeviceConfig("cpu")), model_config)

    assert observed == ["MarlinMxFp4LinearKernel"]


_SELECTOR_PATTERN = re.compile(r"^(?:init|choose|select|make)_.*(?:kernel|backend)$")


def _kernel_selector_names() -> set[str]:
    """Names of the functions that resolve a quantization kernel.

    Discovered from the source rather than an import list, because the MoE
    factories are not re-exported from their package.
    """
    roots = [
        pathlib.Path(vllm.model_executor.kernels.linear.__file__).parent,
        pathlib.Path(vllm.model_executor.layers.fused_moe.oracle.__file__).parent,
        pathlib.Path(
            vllm.model_executor.layers.quantization.utils.humming.__file__
        ).parent,
    ]
    names: set[str] = set()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and _SELECTOR_PATTERN.match(
                    node.name
                ):
                    names.add(node.name)
    return names


def test_kernel_attributes_cover_every_selector_call_site():
    """Guard against a quant method storing its kernel under an unprobed name.

    `model_inspection` recognises a fixed list of attribute names because the
    name differs per quant method. A new method that picks a different name
    would silently report no kernel, so this test derives the ground truth from
    the source instead of trusting the list.
    """
    quantization = pathlib.Path(vllm.model_executor.layers.quantization.__file__).parent
    selectors = _kernel_selector_names()
    assert selectors, "expected to find kernel selectors; imports may have changed"

    probed = set(_KERNEL_ATTRIBUTES) | set(_BACKEND_ATTRIBUTES)
    unprobed: dict[str, str] = {}
    for path in sorted(quantization.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            called = {
                sub.func.id
                for sub in ast.walk(node.value)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            if not called & selectors:
                continue
            # Walk the targets rather than iterating them, so tuple unpacking
            # such as `self.backend, self.experts_cls = select_...()` counts.
            for target in node.targets:
                for sub in ast.walk(target):
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"
                        and sub.attr not in probed
                    ):
                        unprobed[sub.attr] = f"{path.name}:{node.lineno}"

    assert not unprobed, (
        "attributes resolved by a kernel/backend selector are missing from "
        "_KERNEL_ATTRIBUTES / _BACKEND_ATTRIBUTES in vllm/model_inspection.py: "
        f"{unprobed}"
    )
