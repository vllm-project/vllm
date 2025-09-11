# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig

from vllm.config import KVTransferConfig, ModelConfig
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine)


class FakeNCCLLibrary:
    """Mock implementation of NCCL library for testing."""

    def __init__(self, *args, **kwargs):
        self._unique_id_counter = 0
        self._comm_counter = 0

    def ncclGetUniqueId(self):
        """Mock ncclGetUniqueId that returns a unique ID."""
        mock_id = MagicMock()
        mock_id.internal = [self._unique_id_counter] * 8
        self._unique_id_counter += 1
        return mock_id

    def unique_id_from_bytes(self, data):
        """Mock unique_id_from_bytes."""
        mock_id = MagicMock()
        mock_id.internal = list(data)
        return mock_id

    def ncclCommInitRank(self, nranks, unique_id, rank):
        """Mock ncclCommInitRank that returns a mock communicator."""
        mock_comm = MagicMock()
        mock_comm.rank = rank
        mock_comm.nranks = nranks
        self._comm_counter += 1
        return mock_comm

    def ncclSend(self, buffer, count, dtype, dst, comm, stream):
        """Mock ncclSend."""
        pass

    def ncclRecv(self, buffer, count, dtype, src, comm, stream):
        """Mock ncclRecv."""
        pass


class TestP2pNcclEngine:
    """Test cases for P2pNcclEngine."""

    def create_mock_config(self):
        """Create a mock KVTransferConfig for testing."""
        config = MagicMock(spec=KVTransferConfig)
        config.kv_port = 8080
        config.kv_buffer_size = 1e9
        config.kv_connector_extra_config = {
            'http_port': '8080',
            'proxy_ip': '',
            'proxy_port': '',
            'nccl_num_channels': '8',
            'mem_pool_size_gb': '32',
            'send_type': 'PUT_ASYNC',
            'enable_asymmetric_p2p': True,
            'remote_tp_size': 1,
            'remote_pp_size': 1
        }
        config.get_from_extra_config = lambda key, default: \
            config.kv_connector_extra_config.get(key, default)
        return config

    def create_mock_model_config(self):
        """Create a mock ModelConfig for testing."""
        config = MagicMock(spec=ModelConfig)
        config.hf_config = MagicMock(spec=PretrainedConfig)
        config.hf_config.num_hidden_layers = 32
        return config

    def make_engine(self,
                    *,
                    send_type: str | None = None,
                    enable_asymmetric_p2p: bool | None = None,
                    remote_tp_size: int | None = None,
                    remote_pp_size: int | None = None) -> P2pNcclEngine:
        """Unified helper to build a P2pNcclEngine for tests.

        Optional overrides are applied to `kv_connector_extra_config`.
        """
        config = self.create_mock_config()
        if send_type is not None:
            config.kv_connector_extra_config['send_type'] = send_type
        if enable_asymmetric_p2p is not None:
            config.kv_connector_extra_config['enable_asymmetric_p2p'] = \
                enable_asymmetric_p2p
        if remote_tp_size is not None:
            config.kv_connector_extra_config['remote_tp_size'] = remote_tp_size
        if remote_pp_size is not None:
            config.kv_connector_extra_config['remote_pp_size'] = remote_pp_size
        model_config = self.create_mock_model_config()
        return P2pNcclEngine(rank=0,
                             local_rank=0,
                             config=config,
                             model_config=model_config)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_engine_initialization(self, mock_get_ip, mock_get_pp_group,
                                   mock_get_tp_world_size, mock_get_tp_rank,
                                   mock_nccl_lib):
        """Test P2pNcclEngine initialization."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 2
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine()

        assert engine.rank == 0
        assert engine.local_rank == 0
        assert engine.remote_tp_size == 1
        assert engine.remote_pp_size == 1
        assert engine.nccl_num_channels == "8"
        assert engine.send_type == "PUT_ASYNC"

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_get_send_queue_items_non_mla_symmetric(self, mock_get_ip,
                                                    mock_get_pp_group,
                                                    mock_get_tp_world_size,
                                                    mock_get_tp_rank,
                                                    mock_nccl_lib):
        """Test send_tensor method for non-MLA backend with symmetric TP."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(enable_asymmetric_p2p=False)

        # Create test tensor (MLA format)
        tensor = torch.randn(1, 16, 128, dtype=torch.float16)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        result = engine.get_send_queue_items(request_id,
                                             "layers.0",
                                             tensor,
                                             is_mla=True)

        assert len(result) == 1

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_get_send_queue_items_mla_asymmetric_p2p(self, mock_get_ip,
                                                     mock_get_pp_group,
                                                     mock_get_tp_world_size,
                                                     mock_get_tp_rank,
                                                     mock_nccl_lib):
        """Test send_tensor method for MLA backend."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(remote_tp_size=1, remote_pp_size=1)

        # Create test tensor (MLA format)
        tensor = torch.randn(1, 16, 128, dtype=torch.float16)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        result = engine.get_send_queue_items(request_id,
                                             "layers.0",
                                             tensor,
                                             is_mla=True)
        assert len(result) == 1
        expected_addresses = ["10.0.1.3:22001"]
        assert sorted([i.remote_address for i in result]) == \
            sorted(expected_addresses)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_get_send_queue_items_non_mla_with_asymmetric_tp_size(
            self, mock_get_ip, mock_get_pp_group, mock_get_tp_world_size,
            mock_get_tp_rank, mock_nccl_lib):
        """Test send_tensor method for non-MLA backend with TP splitting."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(remote_tp_size=2, remote_pp_size=1)

        # Create test tensor FlashAttention format:
        # (2, num_blocks, block_size, num_heads, head_size)
        tensor = torch.randn(2, 1, 16, 8, 128, dtype=torch.float16)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        result = engine.get_send_queue_items(request_id,
                                             "layers.0",
                                             tensor,
                                             is_mla=False)

        assert isinstance(result, list)
        assert len(result) == 2  # Split into 2 SendQueueItem
        expected_addresses = ["10.0.1.3:22001", "10.0.1.3:22002"]
        assert sorted([i.remote_address for i in result]) == \
            sorted(expected_addresses)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_get_send_queue_items_non_mla_with_asymmetric_tp_size_2_4(
            self, mock_get_ip, mock_get_pp_group, mock_get_tp_world_size,
            mock_get_tp_rank, mock_nccl_lib):
        """Test send_tensor method for non-MLA backend with TP splitting."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 1  # 1st rank
        mock_get_tp_world_size.return_value = 2
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(remote_tp_size=2, remote_pp_size=1)

        # Create test tensor FlashAttention format:
        # (2, num_blocks, block_size, num_heads, head_size)
        tensor = torch.randn(2, 1, 16, 4, 128, dtype=torch.float16)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        result = engine.get_send_queue_items(request_id,
                                             "layers.0",
                                             tensor,
                                             is_mla=False)

        assert isinstance(result, list)
        assert len(result) == 2  # Split into 2 SendQueueItem
        expected_addresses = ["10.0.1.3:22003", "10.0.1.3:22004"]
        assert sorted([i.remote_address for i in result]) == \
            sorted(expected_addresses)
        expected_heads_size = 2
        assert all([i.tensor.shape[3] == expected_heads_size for i in result])

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_get_send_queue_items_non_mla_with_asymmetric_pp_size(
            self, mock_get_ip, mock_get_pp_group, mock_get_tp_world_size,
            mock_get_tp_rank, mock_nccl_lib):
        """Test send_tensor method for non-MLA backend with TP splitting."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 1
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(remote_tp_size=1, remote_pp_size=2)

        # Create test tensor FlashAttention format:
        # (2, num_blocks, block_size, num_heads, head_size)
        tensor = torch.randn(2, 1, 16, 8, 128, dtype=torch.float16)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        result = engine.get_send_queue_items(request_id,
                                             "layers.16",
                                             tensor,
                                             is_mla=False)

        assert isinstance(result, list)
        assert len(result) == 1
        expected_addresses = ["10.0.1.3:22002"]
        assert sorted([i.remote_address for i in result]) == \
            sorted(expected_addresses)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_get_send_queue_items_tp_ratio_assertion(self, mock_get_ip,
                                                     mock_get_pp_group,
                                                     mock_get_tp_world_size,
                                                     mock_get_tp_rank,
                                                     mock_nccl_lib):
        """Test send_tensor assertion for invalid TP ratio."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 2
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(
            remote_tp_size=3,  # Not divisible by 2
            remote_pp_size=1)

        tensor = torch.randn(2, 1, 16, 8, 128, dtype=torch.float16)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        with pytest.raises(AssertionError):
            engine.get_send_queue_items(request_id,
                                        "layers.0",
                                        tensor,
                                        is_mla=False)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_get_send_queue_items_kv_heads_assertion(self, mock_get_ip,
                                                     mock_get_pp_group,
                                                     mock_get_tp_world_size,
                                                     mock_get_tp_rank,
                                                     mock_nccl_lib):
        """Test send_tensor assertion for invalid KV heads ratio."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(remote_tp_size=2, remote_pp_size=1)

        # Create tensor with 7 heads (not divisible by 2)
        tensor = torch.randn(2, 1, 16, 7, 128, dtype=torch.float16)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        with pytest.raises(AssertionError):
            engine.get_send_queue_items(request_id,
                                        "layers.0",
                                        tensor,
                                        is_mla=False)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_recv_tensor_with_put_async(self, mock_get_ip, mock_get_pp_group,
                                        mock_get_tp_world_size,
                                        mock_get_tp_rank, mock_nccl_lib):
        """Test recv_tensor method."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine()

        # Test with tensor in recv_store
        tensor_id = "test_req#layers.0"
        test_tensor = torch.randn(2, 1, 16, 8, 128, dtype=torch.float16)
        engine.recv_store[tensor_id] = test_tensor

        result = engine.recv_tensor("test_req", "layers.0")

        assert torch.equal(result, test_tensor)

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_recv_tensor_with_get_send_type_assertion(self, mock_get_ip,
                                                      mock_get_pp_group,
                                                      mock_get_tp_world_size,
                                                      mock_get_tp_rank,
                                                      mock_nccl_lib):
        """Test recv_tensor method with GET type."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(send_type="GET", remote_tp_size=2)
        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501

        with pytest.raises(NotImplementedError):
            engine.recv_tensor(request_id, "layers.0")

    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.NCCLLibrary"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_rank"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_tensor_model_parallel_world_size"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_pp_group"
    )
    @patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine.get_ip"
    )
    def test_compute_remote_pp_rank(self, mock_get_ip, mock_get_pp_group,
                                    mock_get_tp_world_size, mock_get_tp_rank,
                                    mock_nccl_lib):
        """Test compute_remote_pp_rank method."""
        mock_get_ip.return_value = "127.0.0.1"
        mock_get_tp_rank.return_value = 0
        mock_get_tp_world_size.return_value = 1
        mock_get_pp_group.return_value.rank_in_group = 0
        mock_nccl_lib.return_value = FakeNCCLLibrary()

        engine = self.make_engine(remote_pp_size=2)

        # Test with different layer indices
        assert engine.compute_remote_pp_rank(
            "model.layers.0.self_attn") == 0  # layer 0 -> pp_rank 0
        assert engine.compute_remote_pp_rank(
            "model.layers.16.self_attn") == 1  # layer 16 -> pp_rank 1
        assert engine.compute_remote_pp_rank(
            "model.layers.31.self_attn") == 1  # layer 31 -> pp_rank 1

    def test_get_tensor_id(self):
        """Test get_tensor_id static method."""
        tensor_id = P2pNcclEngine.get_tensor_id("test_req", "layers.0")
        assert tensor_id == "test_req#layers.0"

    def test_parse_request_id(self):
        """Test parse_request_id static method."""

        request_id = "cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0"  # noqa: E501
        ip, port = P2pNcclEngine.parse_request_id(request_id, is_prefill=True)
        assert ip == "10.0.1.3"
        assert port == 22001

        ip, port = P2pNcclEngine.parse_request_id(request_id, is_prefill=False)
        assert ip == "10.0.1.2"
        assert port == 21001

        # Test invalid request_id
        with pytest.raises(
                ValueError,
                match="Request id .* does not contain hostname and port"):
            P2pNcclEngine.parse_request_id("invalid_request", is_prefill=True)
