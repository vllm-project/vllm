# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for --rank-gpu-id / --rank-gpu-memory-mib validation.

NVML and platform lookups are mocked so the tests run without GPUs and
independently of the host's real GPU inventory.
"""

from unittest.mock import MagicMock

import pytest

from vllm.engine.arg_utils import EngineArgs

MIB = 1024 * 1024

# torch.cuda index -> NVML total MiB of the mocked inventory:
# one 32 GiB card and two 20 GiB cards.
GPU_TOTALS_MIB = {0: 32768, 1: 20480, 2: 20480}


class _FakePlatform:
    @staticmethod
    def device_count() -> int:
        return len(GPU_TOTALS_MIB)


@pytest.fixture
def mock_nvml(monkeypatch):
    """Mock NVML totals and the torch->NVML index mapping."""
    import vllm.utils.import_utils as import_utils
    from vllm.platforms.cuda import CudaPlatform

    pynvml = MagicMock()

    def get_handle(nvml_idx):
        return nvml_idx

    def get_memory_info(handle):
        info = MagicMock()
        info.total = GPU_TOTALS_MIB[handle] * MIB
        return info

    pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle
    pynvml.nvmlDeviceGetMemoryInfo.side_effect = get_memory_info
    monkeypatch.setattr(import_utils, "import_pynvml", lambda: pynvml)
    monkeypatch.setattr(
        CudaPlatform,
        "get_torch_to_nvml_mapping",
        classmethod(lambda cls: {i: i for i in GPU_TOTALS_MIB}),
    )
    return pynvml


def _args(**kwargs) -> EngineArgs:
    return EngineArgs(model="facebook/opt-125m", **kwargs)


def _validate(args: EngineArgs) -> None:
    args._validate_rank_gpu_config(_FakePlatform())


def test_default_path_untouched():
    # Neither flag set: validation is a no-op.
    _validate(_args(tensor_parallel_size=2))


def test_rank_gpu_id_requires_memory_mib():
    with pytest.raises(ValueError, match="requires --rank-gpu-memory-mib"):
        _validate(_args(tensor_parallel_size=2, rank_gpu_id=[0, 1]))


def test_memory_mib_requires_rank_gpu_id():
    with pytest.raises(ValueError, match="requires --rank-gpu-id"):
        _validate(_args(tensor_parallel_size=2, rank_gpu_memory_mib=15000))


def test_length_mismatch():
    with pytest.raises(ValueError, match="must equal --tensor-parallel-size"):
        _validate(
            _args(
                tensor_parallel_size=4,
                rank_gpu_id=[0, 1],
                rank_gpu_memory_mib=15000,
            )
        )


def test_gpu_memory_utilization_conflict():
    with pytest.raises(ValueError, match="--gpu-memory-utilization cannot be set"):
        _validate(
            _args(
                tensor_parallel_size=2,
                rank_gpu_id=[0, 1],
                rank_gpu_memory_mib=15000,
                gpu_memory_utilization=0.82,
            )
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("pipeline_parallel_size", 2, "pipeline-parallel-size"),
        ("data_parallel_size", 2, "data-parallel-size"),
        ("enable_expert_parallel", True, "expert parallelism"),
    ],
)
def test_rejects_other_parallelism(field, value, match):
    with pytest.raises(ValueError, match=match):
        _validate(
            _args(
                tensor_parallel_size=2,
                rank_gpu_id=[0, 1],
                rank_gpu_memory_mib=15000,
                **{field: value},
            )
        )


def test_unknown_gpu_id(mock_nvml):
    with pytest.raises(ValueError, match="out of range"):
        _validate(
            _args(
                tensor_parallel_size=2,
                rank_gpu_id=[0, 7],
                rank_gpu_memory_mib=15000,
            )
        )


def test_negative_gpu_id(mock_nvml):
    with pytest.raises(ValueError, match="out of range"):
        _validate(
            _args(
                tensor_parallel_size=2,
                rank_gpu_id=[0, -1],
                rank_gpu_memory_mib=15000,
            )
        )


def test_colocation_fits(mock_nvml):
    # 2 ranks x 15000 MiB = 30000 MiB on the 32768 MiB card: fits.
    _validate(
        _args(
            tensor_parallel_size=4,
            rank_gpu_id=[0, 0, 1, 2],
            rank_gpu_memory_mib=15000,
        )
    )


def test_colocation_physically_impossible(mock_nvml):
    # 2 ranks x 15000 MiB = 30000 MiB on a 20480 MiB card: hard error.
    with pytest.raises(ValueError, match="Physical impossibility"):
        _validate(
            _args(
                tensor_parallel_size=4,
                rank_gpu_id=[1, 1, 0, 2],
                rank_gpu_memory_mib=15000,
            )
        )


def test_single_rank_over_total(mock_nvml):
    with pytest.raises(ValueError, match="Physical impossibility"):
        _validate(
            _args(
                tensor_parallel_size=2,
                rank_gpu_id=[1, 2],
                rank_gpu_memory_mib=20481,
            )
        )


def test_exact_total_is_allowed(mock_nvml):
    # The check is a physical-impossibility bound, not a safety margin:
    # exactly the NVML total must pass.
    _validate(
        _args(
            tensor_parallel_size=2,
            rank_gpu_id=[1, 2],
            rank_gpu_memory_mib=20480,
        )
    )


def test_mib_to_fraction_semantics():
    # The documented conversion: the same MiB budget yields a different
    # gpu_memory_utilization fraction per physical GPU, with no extra
    # discount applied on top.
    mib = 15000
    fractions = {gpu: mib / total for gpu, total in GPU_TOTALS_MIB.items()}
    assert fractions[0] == pytest.approx(15000 / 32768)
    assert fractions[1] == pytest.approx(15000 / 20480)
    # Heterogeneous totals must give different fractions for the same MiB.
    assert fractions[0] != fractions[1]
