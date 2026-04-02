# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import VllmConfig
from vllm.v1.worker.cpu_worker import CPUWorker


def test_cpu_worker_init_device_multi_node():
    """Test CPUWorker.init_device with multi-node DP scenario
    
    This is the critical test case that verifies the fix for the IndexError bug.
    In a 2-node setup with TP=2, DP enabled:
    - Node 0: rank 0, 1 (local_dp_rank=0)
        - Should get omp_cpuids_list[0] and omp_cpuids_list[1]
    - Node 1: rank 2, 3 (local_dp_rank=0)
        - Should get omp_cpuids_list[0] and omp_cpuids_list[1]
        - Without the fix, would fail with IndexError
    """
    # Mock vllm config
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = MagicMock(name='mock.model_config')
    vllm_config.model_config.seed = 42
    vllm_config.cache_config = MagicMock(name='mock.cache_config')
    vllm_config.cache_config.cpu_kvcache_space_bytes = 1024 * 1024 * 1024
    vllm_config.compilation_config = MagicMock(name='mock.compilation_config')
    vllm_config.compilation_config.compilation_time = 0.0
    vllm_config.profiler_config = MagicMock(name='mock.profiler_config')
    vllm_config.profiler_config.profiler = None
    vllm_config.parallel_config.data_parallel_rank_local = 0
    vllm_config.parallel_config.world_size = 2
    vllm_config.parallel_config.disable_custom_all_reduce = False
    vllm_config.weight_transfer_config = MagicMock(
        name='mock.weight_transfer_config')
    vllm_config.weight_transfer_config.backend = "nccl"
    vllm_config.instance_id = "test-instance"
    
    # Test Node 0, rank 0
    with patch.dict(os.environ,
                    {"VLLM_CPU_OMP_THREADS_BIND": "0-15|16-31|32-47|48-63"}), \
         patch("vllm.v1.worker.cpu_worker.init_worker_distributed_environment"), \
         patch("vllm.v1.worker.cpu_worker.set_random_seed"), \
         patch("vllm.v1.worker.cpu_worker.CPUModelRunner"), \
         patch("torch.ops._C.init_cpu_threads_env", create=True):
        worker = CPUWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:12345",
        )
        worker.init_device()
        # local_index = 0 % 2 = 0
        assert worker.local_omp_cpuid == "0-15"
    
    # Test Node 0, rank 1
    with patch.dict(os.environ,
                    {"VLLM_CPU_OMP_THREADS_BIND": "0-15|16-31|32-47|48-63"}), \
         patch("vllm.v1.worker.cpu_worker.init_worker_distributed_environment"), \
         patch("vllm.v1.worker.cpu_worker.set_random_seed"), \
         patch("vllm.v1.worker.cpu_worker.CPUModelRunner"), \
         patch("torch.ops._C.init_cpu_threads_env", create=True):
        worker = CPUWorker(
            vllm_config=vllm_config,
            local_rank=1,
            rank=1,
            distributed_init_method="tcp://127.0.0.1:12345",
        )
        worker.init_device()
        # local_index = 1 % 2 = 1
        assert worker.local_omp_cpuid == "16-31"
    
    # Test Node 1, rank 2 (CRITICAL TEST CASE)
    # This would fail with IndexError before the fix
    with patch.dict(os.environ,
                    {"VLLM_CPU_OMP_THREADS_BIND": "0-15|16-31|32-47|48-63"}), \
         patch("vllm.v1.worker.cpu_worker.init_worker_distributed_environment"), \
         patch("vllm.v1.worker.cpu_worker.set_random_seed"), \
         patch("vllm.v1.worker.cpu_worker.CPUModelRunner"), \
         patch("torch.ops._C.init_cpu_threads_env", create=True):
        worker = CPUWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=2,
            distributed_init_method="tcp://127.0.0.1:12346",
        )
        worker.init_device()
        # local_index = 2 % 2 = 0 (CORRECT!)
        assert worker.local_omp_cpuid == "0-15"
    
    # Test Node 1, rank 3 (CRITICAL TEST CASE)
    # This would fail with IndexError before the fix
    with patch.dict(os.environ,
                    {"VLLM_CPU_OMP_THREADS_BIND": "0-15|16-31|32-47|48-63"}), \
         patch("vllm.v1.worker.cpu_worker.init_worker_distributed_environment"), \
         patch("vllm.v1.worker.cpu_worker.set_random_seed"), \
         patch("vllm.v1.worker.cpu_worker.CPUModelRunner"), \
         patch("torch.ops._C.init_cpu_threads_env", create=True):
        worker = CPUWorker(
            vllm_config=vllm_config,
            local_rank=1,
            rank=3,
            distributed_init_method="tcp://127.0.0.1:12346",
        )
        worker.init_device()
        # local_index = 3 % 2 = 1 (CORRECT!)
        assert worker.local_omp_cpuid == "16-31"


def test_cpu_worker_init_device_misconfigured_non_dp():
    """Test CPUWorker.init_device with misconfigured
    VLLM_CPU_OMP_THREADS_BIND in non-DP mode"""
    # Mock vllm config
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = MagicMock(name='mock.model_config')
    vllm_config.model_config.seed = 42
    vllm_config.cache_config = MagicMock(name='mock.cache_config')
    vllm_config.cache_config.cpu_kvcache_space_bytes = 1024 * 1024 * 1024
    vllm_config.compilation_config = MagicMock(name='mock.compilation_config')
    vllm_config.compilation_config.compilation_time = 0.0
    vllm_config.profiler_config = MagicMock(name='mock.profiler_config')
    vllm_config.profiler_config.profiler = None
    vllm_config.parallel_config.data_parallel_rank_local = None
    vllm_config.parallel_config.world_size = 2
    vllm_config.parallel_config.disable_custom_all_reduce = False
    vllm_config.weight_transfer_config = MagicMock(
        name='mock.weight_transfer_config')
    vllm_config.weight_transfer_config.backend = "nccl"
    vllm_config.instance_id = "test-instance"
    
    # Only 1 entry but rank is 1 (needs at least 2 entries)
    with patch.dict(os.environ, {"VLLM_CPU_OMP_THREADS_BIND": "0-15"}), \
         patch("vllm.v1.worker.cpu_worker.init_worker_distributed_environment"), \
         patch("vllm.v1.worker.cpu_worker.set_random_seed"), \
         patch("vllm.v1.worker.cpu_worker.CPUModelRunner"), \
         patch("torch.ops._C.init_cpu_threads_env", create=True):
        worker = CPUWorker(
            vllm_config=vllm_config,
            local_rank=1,
            rank=1,
            distributed_init_method="tcp://127.0.0.1:12345",
        )
        with pytest.raises(ValueError) as exc_info:
            worker.init_device()
        assert "VLLM_CPU_OMP_THREADS_BIND is misconfigured" in str(
            exc_info.value)
        assert "Expected at least 2 entries" in str(exc_info.value)


def test_cpu_worker_init_device_misconfigured_dp_mode():
    """Test CPUWorker.init_device with misconfigured
    VLLM_CPU_OMP_THREADS_BIND in DP mode"""
    # Mock vllm config
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = MagicMock(name='mock.model_config')
    vllm_config.model_config.seed = 42
    vllm_config.cache_config = MagicMock(name='mock.cache_config')
    vllm_config.cache_config.cpu_kvcache_space_bytes = 1024 * 1024 * 1024
    vllm_config.compilation_config = MagicMock(name='mock.compilation_config')
    vllm_config.compilation_config.compilation_time = 0.0
    vllm_config.profiler_config = MagicMock(name='mock.profiler_config')
    vllm_config.profiler_config.profiler = None
    vllm_config.parallel_config.data_parallel_rank_local = 0
    vllm_config.parallel_config.world_size = 2
    vllm_config.parallel_config.disable_custom_all_reduce = False
    vllm_config.weight_transfer_config = MagicMock(
        name='mock.weight_transfer_config')
    vllm_config.weight_transfer_config.backend = "nccl"
    vllm_config.instance_id = "test-instance"
    
    # Only 1 entry but world_size is 2 (needs at least 2 entries after slicing)
    with patch.dict(os.environ, {"VLLM_CPU_OMP_THREADS_BIND": "0-15"}), \
         patch("vllm.v1.worker.cpu_worker.init_worker_distributed_environment"), \
         patch("vllm.v1.worker.cpu_worker.set_random_seed"), \
         patch("vllm.v1.worker.cpu_worker.CPUModelRunner"), \
         patch("torch.ops._C.init_cpu_threads_env", create=True):
        worker = CPUWorker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:12345",
        )
        with pytest.raises(ValueError) as exc_info:
            worker.init_device()
        assert "VLLM_CPU_OMP_THREADS_BIND is misconfigured" in str(
            exc_info.value)
        assert "Expected at least 2 entries for DP rank 0" in str(
            exc_info.value)
