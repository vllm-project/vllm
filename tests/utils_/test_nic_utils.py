# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig
from vllm.utils import nic_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**parallel_kwargs):
    parallel_defaults = dict(
        nic_bind=False,
        nic_bind_devices=None,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    parallel_defaults.update(parallel_kwargs)
    parallel_config = SimpleNamespace(**parallel_defaults)
    return SimpleNamespace(parallel_config=parallel_config)


# Sample IB device topology for testing
SAMPLE_IB_DEVICES = {
    "mlx5_0": [1],
    "mlx5_1": [1],
    "mlx5_2": [1, 2],
    "mlx5_3": [1],
    "mlx5_10": [1],
    "irdma0": [1],
}


# ---------------------------------------------------------------------------
# validate_nccl_hca_syntax
# ---------------------------------------------------------------------------


class TestValidateNcclHcaSyntax:
    def test_simple_device(self):
        nic_utils.validate_nccl_hca_syntax("mlx5_0:1")

    def test_prefix_pattern(self):
        nic_utils.validate_nccl_hca_syntax("mlx5")

    def test_exact_match(self):
        nic_utils.validate_nccl_hca_syntax("=mlx5_0:1")

    def test_exclude(self):
        nic_utils.validate_nccl_hca_syntax("^mlx5_1")

    def test_exclude_exact(self):
        nic_utils.validate_nccl_hca_syntax("^=mlx5_1:1")

    def test_multi_device(self):
        nic_utils.validate_nccl_hca_syntax("mlx5_0:1,mlx5_1:1")

    def test_mixed_syntax(self):
        nic_utils.validate_nccl_hca_syntax("=mlx5_0:1,^mlx5_1")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            nic_utils.validate_nccl_hca_syntax("")

    def test_double_comma_raises(self):
        with pytest.raises(ValueError, match="empty token"):
            nic_utils.validate_nccl_hca_syntax("mlx5_0:1,,mlx5_1:1")

    def test_bad_chars_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            nic_utils.validate_nccl_hca_syntax("mlx5_0:1;drop")

    def test_double_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            nic_utils.validate_nccl_hca_syntax("mlx5_0:1:2")

    def test_equals_only_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            nic_utils.validate_nccl_hca_syntax("=")

    def test_caret_only_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            nic_utils.validate_nccl_hca_syntax("^")


# ---------------------------------------------------------------------------
# expand_nccl_hca_pattern
# ---------------------------------------------------------------------------


class TestExpandNcclHcaPattern:
    def test_prefix_match(self):
        result = nic_utils.expand_nccl_hca_pattern("mlx5", SAMPLE_IB_DEVICES)
        assert result == [
            "mlx5_0:1",
            "mlx5_1:1",
            "mlx5_10:1",
            "mlx5_2:1",
            "mlx5_2:2",
            "mlx5_3:1",
        ]

    def test_exact_match(self):
        # =mlx5_1 should NOT match mlx5_10
        result = nic_utils.expand_nccl_hca_pattern("=mlx5_1", SAMPLE_IB_DEVICES)
        assert result == ["mlx5_1:1"]

    def test_prefix_match_hits_mlx5_10(self):
        # mlx5_1 (prefix) should match mlx5_1 AND mlx5_10
        result = nic_utils.expand_nccl_hca_pattern("mlx5_1", SAMPLE_IB_DEVICES)
        assert result == ["mlx5_1:1", "mlx5_10:1"]

    def test_with_port(self):
        result = nic_utils.expand_nccl_hca_pattern("mlx5_2:1", SAMPLE_IB_DEVICES)
        assert result == ["mlx5_2:1"]

    def test_with_port_specific(self):
        result = nic_utils.expand_nccl_hca_pattern("mlx5_2:2", SAMPLE_IB_DEVICES)
        assert result == ["mlx5_2:2"]

    def test_exclude_prefix(self):
        result = nic_utils.expand_nccl_hca_pattern("^mlx5_1", SAMPLE_IB_DEVICES)
        # Excludes mlx5_1 and mlx5_10 (prefix match); keeps others
        assert "mlx5_1:1" not in result
        assert "mlx5_10:1" not in result
        assert "mlx5_0:1" in result
        assert "irdma0:1" in result

    def test_exclude_exact(self):
        result = nic_utils.expand_nccl_hca_pattern("^=mlx5_1", SAMPLE_IB_DEVICES)
        # Excludes only mlx5_1, NOT mlx5_10
        assert "mlx5_1:1" not in result
        assert "mlx5_10:1" in result
        assert "mlx5_0:1" in result

    def test_include_and_exclude(self):
        result = nic_utils.expand_nccl_hca_pattern("mlx5,^=mlx5_3", SAMPLE_IB_DEVICES)
        assert "mlx5_3:1" not in result
        assert "mlx5_0:1" in result
        assert "irdma0:1" not in result

    def test_no_match_raises(self):
        with pytest.raises(RuntimeError, match="matched no devices"):
            nic_utils.expand_nccl_hca_pattern("nonexistent", SAMPLE_IB_DEVICES)

    def test_multi_device_comma(self):
        result = nic_utils.expand_nccl_hca_pattern(
            "=mlx5_0:1,=mlx5_3:1", SAMPLE_IB_DEVICES
        )
        assert result == ["mlx5_0:1", "mlx5_3:1"]


# ---------------------------------------------------------------------------
# enumerate_ib_devices (with monkeypatched sysfs)
# ---------------------------------------------------------------------------


class TestEnumerateIbDevices:
    def test_no_sysfs_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", tmp_path / "noexist")
        nic_utils.enumerate_ib_devices.cache_clear()
        assert nic_utils.enumerate_ib_devices() is None
        nic_utils.enumerate_ib_devices.cache_clear()

    def test_enumerates_devices(self, monkeypatch, tmp_path):
        ib_path = tmp_path / "infiniband"
        ib_path.mkdir()
        (ib_path / "mlx5_0" / "ports" / "1").mkdir(parents=True)
        (ib_path / "mlx5_1" / "ports" / "1").mkdir(parents=True)
        (ib_path / "mlx5_1" / "ports" / "2").mkdir(parents=True)

        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        nic_utils.enumerate_ib_devices.cache_clear()
        result = nic_utils.enumerate_ib_devices()
        nic_utils.enumerate_ib_devices.cache_clear()

        assert result == {"mlx5_0": [1], "mlx5_1": [1, 2]}

    def test_empty_dir_returns_none(self, monkeypatch, tmp_path):
        ib_path = tmp_path / "infiniband"
        ib_path.mkdir()

        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        nic_utils.enumerate_ib_devices.cache_clear()
        assert nic_utils.enumerate_ib_devices() is None
        nic_utils.enumerate_ib_devices.cache_clear()


# ---------------------------------------------------------------------------
# get_ib_device_numa_node
# ---------------------------------------------------------------------------


class TestGetIbDeviceNumaNode:
    def test_reads_numa_node(self, monkeypatch, tmp_path):
        ib_path = tmp_path / "infiniband"
        dev_path = ib_path / "mlx5_0" / "device"
        dev_path.mkdir(parents=True)
        (dev_path / "numa_node").write_text("1\n")

        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        assert nic_utils.get_ib_device_numa_node("mlx5_0") == 1

    def test_negative_one_returns_none(self, monkeypatch, tmp_path):
        ib_path = tmp_path / "infiniband"
        dev_path = ib_path / "mlx5_0" / "device"
        dev_path.mkdir(parents=True)
        (dev_path / "numa_node").write_text("-1\n")

        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        assert nic_utils.get_ib_device_numa_node("mlx5_0") is None

    def test_missing_file_returns_none(self, monkeypatch, tmp_path):
        ib_path = tmp_path / "infiniband"
        (ib_path / "mlx5_0" / "device").mkdir(parents=True)

        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        assert nic_utils.get_ib_device_numa_node("mlx5_0") is None


# ---------------------------------------------------------------------------
# get_nic_env_vars
# ---------------------------------------------------------------------------


class TestGetNicEnvVars:
    def test_disabled_returns_none(self):
        vllm_config = _make_config(nic_bind=False)
        assert nic_utils.get_nic_env_vars(vllm_config, local_rank=0) is None

    def test_explicit_devices(self, monkeypatch, tmp_path):
        # Set up fake sysfs for UCX expansion
        ib_path = tmp_path / "infiniband"
        (ib_path / "mlx5_0" / "ports" / "1").mkdir(parents=True)
        (ib_path / "mlx5_1" / "ports" / "1").mkdir(parents=True)
        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        nic_utils.enumerate_ib_devices.cache_clear()

        vllm_config = _make_config(
            nic_bind=True,
            nic_bind_devices=["=mlx5_0:1", "=mlx5_1:1"],
        )
        env = nic_utils.get_nic_env_vars(vllm_config, local_rank=0)
        assert env is not None
        assert env["NCCL_IB_HCA"] == "=mlx5_0:1"
        assert env["UCX_NET_DEVICES"] == "mlx5_0:1"

        env1 = nic_utils.get_nic_env_vars(vllm_config, local_rank=1)
        assert env1 is not None
        assert env1["NCCL_IB_HCA"] == "=mlx5_1:1"
        assert env1["UCX_NET_DEVICES"] == "mlx5_1:1"
        nic_utils.enumerate_ib_devices.cache_clear()

    def test_gpu_index_out_of_range(self):
        vllm_config = _make_config(nic_bind=True, nic_bind_devices=["mlx5_0:1"])
        with pytest.raises(ValueError, match="exceeds nic_bind_devices"):
            nic_utils.get_nic_env_vars(vllm_config, local_rank=1)


# ---------------------------------------------------------------------------
# configure_subprocess context manager
# ---------------------------------------------------------------------------


class TestConfigureSubprocess:
    def test_sets_and_restores_env(self, monkeypatch, tmp_path):
        ib_path = tmp_path / "infiniband"
        (ib_path / "mlx5_0" / "ports" / "1").mkdir(parents=True)
        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        nic_utils.enumerate_ib_devices.cache_clear()

        vllm_config = _make_config(nic_bind=True, nic_bind_devices=["=mlx5_0:1"])

        # Ensure env vars are not set initially
        os.environ.pop("NCCL_IB_HCA", None)
        os.environ.pop("UCX_NET_DEVICES", None)

        with nic_utils.configure_subprocess(vllm_config, local_rank=0):
            assert os.environ["NCCL_IB_HCA"] == "=mlx5_0:1"
            assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1"

        assert "NCCL_IB_HCA" not in os.environ
        assert "UCX_NET_DEVICES" not in os.environ
        nic_utils.enumerate_ib_devices.cache_clear()

    def test_restores_previous_values(self, monkeypatch, tmp_path):
        ib_path = tmp_path / "infiniband"
        (ib_path / "mlx5_0" / "ports" / "1").mkdir(parents=True)
        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        nic_utils.enumerate_ib_devices.cache_clear()

        os.environ["NCCL_IB_HCA"] = "old-hca"
        os.environ["UCX_NET_DEVICES"] = "old-ucx"

        vllm_config = _make_config(nic_bind=True, nic_bind_devices=["=mlx5_0:1"])

        with nic_utils.configure_subprocess(vllm_config, local_rank=0):
            assert os.environ["NCCL_IB_HCA"] == "=mlx5_0:1"
            assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1"

        assert os.environ["NCCL_IB_HCA"] == "old-hca"
        assert os.environ["UCX_NET_DEVICES"] == "old-ucx"

        # Cleanup
        os.environ.pop("NCCL_IB_HCA", None)
        os.environ.pop("UCX_NET_DEVICES", None)
        nic_utils.enumerate_ib_devices.cache_clear()

    def test_noop_when_disabled(self):
        vllm_config = _make_config(nic_bind=False)
        with nic_utils.configure_subprocess(vllm_config, local_rank=0):
            pass  # should not raise


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_get_auto_nic_devices(self, monkeypatch, tmp_path):
        # Set up fake sysfs: mlx5_0 on NUMA 0, mlx5_1 on NUMA 1
        ib_path = tmp_path / "infiniband"
        for dev, numa_node in [("mlx5_0", 0), ("mlx5_1", 1)]:
            dev_path = ib_path / dev
            (dev_path / "ports" / "1").mkdir(parents=True)
            (dev_path / "device").mkdir(parents=True, exist_ok=True)
            (dev_path / "device" / "numa_node").write_text(f"{numa_node}\n")

        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", ib_path)
        nic_utils.enumerate_ib_devices.cache_clear()
        nic_utils.get_auto_nic_devices.cache_clear()

        # Mock _get_gpu_numa_nodes to return a known GPU-to-NUMA mapping
        monkeypatch.setattr(nic_utils, "_get_gpu_numa_nodes", lambda: [0, 1])

        result = nic_utils.get_auto_nic_devices()
        # GPU 0 on NUMA 0 → =mlx5_0:1, GPU 1 on NUMA 1 → =mlx5_1:1
        assert result == ["=mlx5_0:1", "=mlx5_1:1"]

        nic_utils.enumerate_ib_devices.cache_clear()
        nic_utils.get_auto_nic_devices.cache_clear()

    def test_get_auto_nic_devices_no_gpu_numa(self, monkeypatch):
        monkeypatch.setattr(nic_utils, "_get_gpu_numa_nodes", lambda: None)
        nic_utils.get_auto_nic_devices.cache_clear()
        assert nic_utils.get_auto_nic_devices() is None
        nic_utils.get_auto_nic_devices.cache_clear()

    def test_get_auto_nic_devices_no_ib(self, monkeypatch, tmp_path):
        monkeypatch.setattr(nic_utils, "_get_gpu_numa_nodes", lambda: [0, 1])
        monkeypatch.setattr(nic_utils, "_SYSFS_IB_PATH", tmp_path / "noexist")
        nic_utils.enumerate_ib_devices.cache_clear()
        nic_utils.get_auto_nic_devices.cache_clear()
        assert nic_utils.get_auto_nic_devices() is None
        nic_utils.enumerate_ib_devices.cache_clear()
        nic_utils.get_auto_nic_devices.cache_clear()


# ---------------------------------------------------------------------------
# ParallelConfig validation
# ---------------------------------------------------------------------------


class TestParallelConfigValidation:
    def test_nic_bind_devices_requires_nic_bind(self):
        with pytest.raises(ValueError, match="requires nic_bind=True"):
            ParallelConfig(nic_bind_devices=["mlx5_0:1"])

    def test_rejects_invalid_syntax(self):
        with pytest.raises(ValueError, match="Invalid"):
            ParallelConfig(nic_bind=True, nic_bind_devices=["mlx5_0:1:2"])

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ParallelConfig(nic_bind=True, nic_bind_devices=[])

    def test_accepts_valid_devices(self):
        config = ParallelConfig(
            nic_bind=True,
            nic_bind_devices=["=mlx5_0:1", "mlx5_1:1"],
        )
        assert config.nic_bind_devices == ["=mlx5_0:1", "mlx5_1:1"]
