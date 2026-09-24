# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Model Runner V2 pluggable components (Layer 2).

The V2 model runner fetches its internal component classes from
``Platform.get_runner_component``. These tests pin that contract:
the default returns the GPU classes, and a backend can substitute individual
pieces with ``dataclasses.replace``.
"""

from dataclasses import replace

from vllm.platforms import current_platform
from vllm.platforms.interface import Platform, RunnerComponents
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.ubatch_utils import maybe_build_ubatch_runner


def test_default_runner_component_cls():
    classes = Platform().get_runner_component()
    assert classes == RunnerComponents(
        input_batch_cls=InputBatch,
        cudagraph_manager_cls=ModelCudaGraphManager,
        sampler_cls=Sampler,
        pcp_manager_cls=PCPManager,
        block_tables_cls=BlockTables,
        rejection_sampler_cls=RejectionSampler,
        pooling_runner_cls=PoolingRunner,
        speculator_factory=init_speculator,
        model_state_factory=init_model_state,
        ubatch_runner_factory=maybe_build_ubatch_runner,
    )


def test_current_platform_v2_runner_component_cls():
    # The GPU model runner fetches from the active platform.
    classes = current_platform.get_runner_component()
    assert classes.input_batch_cls is InputBatch
    assert classes.cudagraph_manager_cls is ModelCudaGraphManager
    assert classes.sampler_cls is Sampler
    assert classes.pcp_manager_cls is PCPManager
    assert classes.block_tables_cls is BlockTables
    assert classes.rejection_sampler_cls is RejectionSampler
    assert classes.pooling_runner_cls is PoolingRunner
    assert classes.speculator_factory is init_speculator
    assert classes.model_state_factory is init_model_state
    assert classes.ubatch_runner_factory is maybe_build_ubatch_runner


def test_backend_overrides_single_component():
    """A backend substitutes only the pieces that differ."""

    class BackendSampler(Sampler):
        pass

    base = Platform().get_runner_component()
    assert isinstance(base, RunnerComponents)
    overridden = replace(base, sampler_cls=BackendSampler)
    assert overridden.sampler_cls is BackendSampler
    # Everything else keeps the GPU default.
    assert overridden.input_batch_cls is InputBatch
    assert overridden.cudagraph_manager_cls is ModelCudaGraphManager
    assert overridden.pcp_manager_cls is PCPManager
    assert overridden.block_tables_cls is BlockTables
    assert overridden.rejection_sampler_cls is RejectionSampler
    assert overridden.pooling_runner_cls is PoolingRunner
    assert overridden.speculator_factory is init_speculator
    assert overridden.model_state_factory is init_model_state
    assert overridden.ubatch_runner_factory is maybe_build_ubatch_runner


def test_watermark_sampler_composes_over_pluggable_sampler_cls():
    """Watermarking must wrap a backend sampler, not bypass it."""
    from vllm.v1.watermarking.gpu_sampler import (
        GPUWatermarkSampler,
        _watermark_sampler_cls,
    )

    class BackendSampler(Sampler):
        pass

    composed = _watermark_sampler_cls(BackendSampler)
    assert issubclass(composed, GPUWatermarkSampler)
    assert issubclass(composed, BackendSampler)
    # Watermarking takes precedence in the MRO: it layers over the backend.
    assert composed.__mro__.index(GPUWatermarkSampler) < composed.__mro__.index(
        BackendSampler
    )
    # Cached: same base class returns the same composed class.
    assert _watermark_sampler_cls(BackendSampler) is composed
    assert _watermark_sampler_cls(Sampler) is not composed
