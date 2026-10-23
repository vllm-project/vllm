# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the oracle helpers shared by every MoE backend: the
`MoEKernelOracle` selection helpers and `make_moe_kernel`.

Backend policy (platform gates, env vars, activation preferences) and the
explicit-backend mapping of INT8/MXFP4/MXFP8/b12x-NVFP4/AITER-unquantized are
covered in the per-oracle tests; only the uncovered cases are exercised here.
No MoE kernel is launched and nothing is platform specific: kernel probing and
prepare/finalize construction are stubbed out.
"""

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsW4A8Fp8,
)
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    Fp8MoEKernelOracle,
    make_fp8_moe_kernel,
    map_fp8_backend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
    Int8MoEKernelOracle,
    make_int8_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    WNA16MoEKernelOracle,
    map_wna16_backend,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    Mxfp4MoEKernelOracle,
    make_mxfp4_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    _BACKEND_NAME_MAP,
    Mxfp8MoEKernelOracle,
    select_mxfp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    NvFp4MoEKernelOracle,
    make_nvfp4_moe_kernel,
    map_nvfp4_backend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    UnquantizedMoEKernelOracle,
    make_unquantized_moe_kernel,
    map_unquantized_backend,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8 import (
    W4A8MoeBackend,
    make_w4a8_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8_int8 import (
    W4A8Int8MoeBackend,
    W4A8Int8MoEKernelOracle,
    map_w4a8_int8_backend,
    select_w4a8_int8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static
from vllm.platforms import current_platform

# oracle -> (select call, backend mapper, explicit runner backend, name used in
# the "moe_backend=... is not supported for <name> MoE" error).
SELECT_CASES = {
    "fp8": (
        lambda cfg: select_fp8_moe_backend(cfg, None, None),
        map_fp8_backend,
        "triton",
        "FP8",
    ),
    "unquantized": (
        select_unquantized_moe_backend,
        map_unquantized_backend,
        "triton",
        "unquantized",
    ),
    "nvfp4": (
        lambda cfg: select_nvfp4_moe_backend(cfg, None, None),
        map_nvfp4_backend,
        "marlin",
        "NvFP4",
    ),
    "int_wna16": (
        lambda cfg: select_wna16_moe_backend(cfg, kInt4Static, object(), False, False),
        map_wna16_backend,
        "triton",
        "WNA16",
    ),
    "mxfp8": (
        select_mxfp8_moe_backend,
        _BACKEND_NAME_MAP.__getitem__,
        "marlin",
        "MXFP8",
    ),
    "w4a8_int8": (
        lambda cfg: select_w4a8_int8_moe_backend(cfg, None, None),
        map_w4a8_int8_backend,
        "cpu",
        "W4A8 Int8",
    ),
}

# oracle -> (make_*_moe_kernel call, experts class, expected prepare/finalize
# extras, expected experts kwargs, optional experts log template).
ASSEMBLY_CASES = {
    "fp8": (
        lambda quant_config, moe_config, experts_cls: make_fp8_moe_kernel(
            quant_config, moe_config, experts_cls, Fp8MoeBackend.FLASHINFER_CUTLASS
        ),
        TritonExperts,
        {"input_dtype": torch.bfloat16},
        {},
        None,
    ),
    "int8": (
        lambda quant_config, moe_config, experts_cls: make_int8_moe_kernel(
            Int8MoeBackend.TRITON, quant_config, moe_config, experts_cls
        ),
        TritonExperts,
        {},
        {},
        None,
    ),
    "mxfp4": (
        lambda quant_config, moe_config, experts_cls: make_mxfp4_moe_kernel(
            quant_config,
            moe_config,
            experts_cls,
            Mxfp4MoeBackend.CUTLASS_MXFP4_MXFP4,
        ),
        TritonExperts,
        {},
        {},
        "Using %s",
    ),
    "nvfp4": (
        lambda quant_config, moe_config, experts_cls: make_nvfp4_moe_kernel(
            quant_config,
            moe_config,
            experts_cls,
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            per_token_activation=True,
        ),
        TritonExperts,
        {"input_dtype": torch.bfloat16},
        {"per_token_activation": True},
        None,
    ),
    "unquantized": (
        lambda quant_config, moe_config, experts_cls: make_unquantized_moe_kernel(
            quant_config, moe_config, UnquantizedMoeBackend.TRITON, experts_cls
        ),
        TritonExperts,
        {},
        {},
        "Using %s MoE backend",
    ),
    "w4a8": (
        lambda quant_config, moe_config, experts_cls: make_w4a8_moe_kernel(
            W4A8MoeBackend.CUTLASS,
            quant_config,
            moe_config,
            experts_cls,
            torch.zeros(1),
            torch.zeros(1),
            128,
        ),
        CutlassExpertsW4A8Fp8,
        {},
        {"group_size": 128},
        None,
    ),
}


@pytest.mark.parametrize(
    ("supported", "reason", "available_backends", "expected_log", "expected_error"),
    [
        pytest.param(
            0,
            None,
            [Int8MoeBackend.TRITON, Int8MoeBackend.HUMMING],
            "Using TRITON Int8 MoE backend out of potential backends: "
            "['TRITON', 'HUMMING'].",
            None,
            id="first-supported-lists-candidates",
        ),
        pytest.param(
            1,
            "not for this deployment",
            None,
            "Using TRITON Int8 MoE backend.",
            None,
            id="second-supported",
        ),
        pytest.param(
            None,
            "rejected by the kernel",
            None,
            None,
            "since rejected by the kernel",
            id="unsupported-with-reason",
        ),
        pytest.param(
            None,
            None,
            None,
            None,
            "does not support the deployment configuration.",
            id="unsupported-without-reason",
        ),
    ],
)
def test_return_or_raise(
    supported, reason, available_backends, expected_log, expected_error, monkeypatch
):
    """Picks the first supported kernel class (logging the unified message) or
    raises `ValueError` with the last rejection reason."""
    logger = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.base.logger", logger
    )
    kernel_classes = [TritonExperts, BatchedTritonExperts]
    for index, kernel_class in enumerate(kernel_classes):
        monkeypatch.setattr(
            kernel_class,
            "is_supported_config",
            staticmethod(
                lambda *_, supported_here=index == supported, why=reason: (
                    supported_here,
                    why,
                )
            ),
        )
    oracle = Int8MoEKernelOracle(
        make_dummy_moe_config(),
        None,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )

    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            oracle.return_or_raise(
                Int8MoeBackend.TRITON, available_backends, kernel_classes
            )
        assert logger.info_once.call_args_list == []
        return

    backend, kernel_class = oracle.return_or_raise(
        Int8MoeBackend.TRITON, available_backends, kernel_classes
    )
    assert backend is Int8MoeBackend.TRITON
    assert kernel_class is kernel_classes[supported]
    assert logger.info_once.call_args_list == [call(expected_log)]


# oracle class -> a backend it maps to kernel classes for.
TRY_BACKEND_CASES = [
    pytest.param(Fp8MoEKernelOracle, Fp8MoeBackend.TRITON, id="fp8"),
    pytest.param(Int8MoEKernelOracle, Int8MoeBackend.TRITON, id="int8"),
    pytest.param(WNA16MoEKernelOracle, WNA16MoEBackend.TRITON, id="int_wna16"),
    pytest.param(Mxfp4MoEKernelOracle, Mxfp4MoeBackend.MARLIN, id="mxfp4"),
    pytest.param(NvFp4MoEKernelOracle, NvFp4MoeBackend.MARLIN, id="nvfp4"),
    pytest.param(
        UnquantizedMoEKernelOracle, UnquantizedMoeBackend.TRITON, id="unquantized"
    ),
    pytest.param(W4A8Int8MoEKernelOracle, W4A8Int8MoeBackend.CPU_INT4, id="w4a8_int8"),
    pytest.param(Mxfp8MoEKernelOracle, Fp8MoeBackend.MARLIN, id="mxfp8"),
]


@pytest.mark.parametrize(("oracle_class", "backend"), TRY_BACKEND_CASES)
def test_try_backend(oracle_class, backend, monkeypatch):
    """Each oracle class carries its own backend -> kernel-class mapping, and
    `try_backend` probes exactly that list: first supported wins, otherwise the
    last rejection reason, with an optional caller-supplied list (mxfp4's
    LoRA-aware candidates)."""
    oracle = oracle_class(make_dummy_moe_config())
    candidates = oracle.backend_to_kernel_cls(backend)
    assert candidates
    assert all(issubclass(candidate, mk.FusedMoEExperts) for candidate in candidates)

    for candidate in candidates:
        rejection = f"{candidate.__name__} rejected"
        monkeypatch.setattr(
            candidate,
            "is_supported_config",
            staticmethod(lambda *_, why=rejection: (False, why)),
        )
    assert oracle.try_backend(backend) == (None, f"{candidates[-1].__name__} rejected")

    monkeypatch.setattr(
        candidates[-1], "is_supported_config", staticmethod(lambda *_: (True, None))
    )
    assert oracle.try_backend(backend) == (candidates[-1], None)

    monkeypatch.setattr(
        TritonExperts, "is_supported_config", staticmethod(lambda *_: (True, None))
    )
    assert oracle.try_backend(backend, [TritonExperts]) == (TritonExperts, None)


@pytest.mark.parametrize("unknown", [False, True], ids=["explicit", "unknown"])
@pytest.mark.parametrize("oracle", SELECT_CASES)
def test_explicit_moe_backend(oracle, unknown, monkeypatch):
    """`--moe-backend=<name>` reaches the shared selection helper with the
    mapped backend, or raises for an unknown name."""
    select, map_backend, runner_backend, oracle_name = SELECT_CASES[oracle]
    if oracle == "w4a8_int8":
        # The W4A8 INT8 oracle only exposes backends on CPU.
        monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    moe_config = make_dummy_moe_config()
    moe_config.moe_backend = "not-a-backend" if unknown else runner_backend

    if unknown:
        with pytest.raises(ValueError, match=f"not supported for {oracle_name} MoE"):
            select(moe_config)
        return

    expected_backend = map_backend(runner_backend)
    try:
        selected_backend, selected_experts_class = select(moe_config)
    except ValueError as error:
        # Unsupported on this platform: the mapped backend must be the rejected
        # one, i.e. the override was not silently dropped or remapped.
        assert f"'{expected_backend.value}'" in str(error)
    else:
        assert selected_backend is expected_backend
        assert selected_experts_class is not None


@pytest.mark.parametrize("batched", [False, True], ids=["standard", "batched"])
@pytest.mark.parametrize("mismatched", [False, True], ids=["matching", "mismatched"])
@pytest.mark.parametrize("oracle", ASSEMBLY_CASES)
def test_make_moe_kernel_assembly(oracle, mismatched, batched, monkeypatch):
    """All oracles assemble prepare/finalize + experts through the shared
    helper: same contract, per-oracle extras preserved."""
    (
        make_kernel,
        experts_cls,
        expected_prepare_finalize_kwargs,
        expected_experts_kwargs,
        experts_log,
    ) = ASSEMBLY_CASES[oracle]
    quant_config = SimpleNamespace(is_block_quantized=True)
    moe_config = make_dummy_moe_config()
    prepare_finalize_kwargs: dict = {}
    experts_kwargs: dict = {}
    sentinel = object()
    logger = Mock()
    prepare_finalize = SimpleNamespace(
        activation_format=(
            mk.FusedMoEActivationFormat.BatchedExperts
            if batched
            else mk.FusedMoEActivationFormat.Standard
        ),
        max_num_tokens_per_rank=lambda: 64,
        num_dispatchers=lambda: 3,
    )

    def fake_prepare_finalize(**kwargs):
        prepare_finalize_kwargs.update(kwargs)
        return prepare_finalize

    def record_experts_init(self, **kwargs):
        experts_kwargs.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.base.maybe_make_prepare_finalize",
        fake_prepare_finalize,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.base.logger", logger
    )
    monkeypatch.setattr(
        mk, "FusedMoEKernel", lambda prepare_finalize, experts: sentinel
    )
    monkeypatch.setattr(experts_cls, "__init__", record_experts_init)
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4."
        "_validate_mxfp4_experts_config",
        lambda *args, **kwargs: None,
    )
    experts_format = (
        mk.FusedMoEActivationFormat.Standard
        if batched
        else mk.FusedMoEActivationFormat.BatchedExperts
    )
    if not mismatched:
        experts_format = prepare_finalize.activation_format
    monkeypatch.setattr(
        experts_cls, "activation_format", staticmethod(lambda: experts_format)
    )

    if mismatched:
        with pytest.raises(ValueError, match="expects .* activations"):
            make_kernel(quant_config, moe_config, experts_cls)
        return

    assert make_kernel(quant_config, moe_config, experts_cls) is sentinel

    assert prepare_finalize_kwargs["moe"] is moe_config
    assert prepare_finalize_kwargs["quant_config"] is quant_config
    assert prepare_finalize_kwargs["routing_tables"] is None
    assert prepare_finalize_kwargs["allow_new_interface"] is True
    assert prepare_finalize_kwargs["use_monolithic"] is False
    assert prepare_finalize_kwargs.get("input_dtype") == (
        expected_prepare_finalize_kwargs.get("input_dtype")
    )

    actual_experts_kwargs = {
        key: value
        for key, value in experts_kwargs.items()
        if key
        not in ("moe_config", "quant_config", "max_num_tokens", "num_dispatchers")
    }
    if oracle == "w4a8":
        # The stride tensors are opaque here; only their presence matters.
        assert isinstance(actual_experts_kwargs.pop("b_strides1"), torch.Tensor)
        assert isinstance(actual_experts_kwargs.pop("b_strides2"), torch.Tensor)
    assert actual_experts_kwargs == expected_experts_kwargs
    assert experts_kwargs.get("max_num_tokens") == (64 if batched else None)
    assert experts_kwargs.get("num_dispatchers") == (3 if batched else None)

    expected_logs = [call("Using %s", "SimpleNamespace")]
    if experts_log is not None:
        expected_logs.append(call(experts_log, experts_cls.__name__))
    assert logger.info_once.call_args_list == expected_logs


def test_make_kernel_delegates() -> None:
    """`UnquantizedMoEKernelOracle.make_kernel` delegates to the module fn."""
    quant_config = object()
    moe_config = object()
    sentinel_kernel = object()
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.unquantized."
        "make_unquantized_moe_kernel",
        return_value=sentinel_kernel,
    ) as mocked:
        built_kernel = UnquantizedMoEKernelOracle(make_dummy_moe_config()).make_kernel(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            TritonExperts,
        )

    mocked.assert_called_once_with(
        quant_config,
        moe_config,
        UnquantizedMoeBackend.TRITON,
        TritonExperts,
        None,  # routing_tables default
    )
    assert built_kernel is sentinel_kernel
