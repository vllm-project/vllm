# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config import ParallelConfig
from vllm.config import parallel as parallel_config_module
from vllm.engine.arg_utils import get_kwargs
from vllm.model_executor.layers.fused_moe import all2all_utils
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareFinalizeFactory,
)


def test_register_prepare_finalize_factory(monkeypatch):
    backend = "test_backend"
    expected = object()
    expected_manager = object()
    monkeypatch.setattr(all2all_utils, "_PREPARE_FINALIZE_FACTORIES", {})
    monkeypatch.setattr(
        parallel_config_module,
        "SEQUENCE_PARALLEL_MOE_BACKENDS",
        set(),
    )
    monkeypatch.setattr(
        all2all_utils,
        "get_ep_all2all_manager",
        lambda *args: expected_manager,
    )

    @all2all_utils.register_moe_prepare_finalize_factory
    class TestPrepareFinalizeFactory(MoEPrepareFinalizeFactory):
        backend_name = backend
        supports_sequence_parallel = True

        @classmethod
        def create(
            cls,
            *,
            moe,
            quant_config,
            routing_tables,
            all2all_manager,
            allow_new_interface,
            use_monolithic,
            eep_stage,
        ):
            assert all2all_manager is expected_manager
            return expected

    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_all2all_kernels=True,
            all2all_backend=backend,
        )
    )

    actual = all2all_utils.maybe_make_prepare_finalize(
        moe=moe,
        quant_config=None,
        routing_tables=None,
        allow_new_interface=True,
    )

    assert actual is expected
    config = ParallelConfig(
        all2all_backend=backend,
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        data_parallel_size=2,
    )
    assert config.use_sequence_parallel_moe


def test_parallel_config_accepts_oot_all2all_backend():
    config = ParallelConfig(all2all_backend="test_oot_prepare_finalize")
    assert config.all2all_backend == "test_oot_prepare_finalize"

    cli_kwargs = get_kwargs(ParallelConfig)["all2all_backend"]
    assert cli_kwargs["type"] is str
    assert "choices" not in cli_kwargs
    assert "allgather_reducescatter" in cli_kwargs["metavar"]
