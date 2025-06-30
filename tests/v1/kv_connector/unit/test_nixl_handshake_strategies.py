# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    HandshakeStrategy, HttpHandshakeStrategy, NixlAgentMetadata,
    ZmqHandshakeStrategy)


class TestHandshakeStrategyAbstraction:

    def test_abstract_base_class(self):
        with pytest.raises(TypeError):
            HandshakeStrategy(None, 0, 1, 8080, "test-engine")


class TestZmqHandshakeStrategy:

    def create_test_metadata(self) -> NixlAgentMetadata:
        return NixlAgentMetadata(engine_id="test-engine",
                                 agent_metadata=b"test-agent-data",
                                 kv_caches_base_addr=[12345],
                                 num_blocks=100,
                                 block_len=16,
                                 attn_backend_name="FLASH_ATTN_VLLM_V1")

    @patch(
        'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.ZmqHandshakeStrategy._zmq_ctx'
    )
    @patch('vllm.utils.make_zmq_path')
    def test_zmq_handshake_success(self, mock_make_path, mock_zmq_ctx):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock(return_value="agent-name-0")

        strategy = ZmqHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                        mock_add_agent)

        mock_socket = MagicMock()
        mock_zmq_ctx.return_value.__enter__.return_value = mock_socket
        mock_make_path.return_value = "tcp://localhost:8080"

        test_metadata = self.create_test_metadata()
        with patch('msgspec.msgpack.Decoder') as mock_decoder_class:
            mock_decoder = MagicMock()
            mock_decoder_class.return_value = mock_decoder
            mock_decoder.decode.return_value = test_metadata

            result = strategy.initiate_handshake("localhost", 8080, 1)

            assert result == {0: "agent-name-0"}
            mock_add_agent.assert_called_once()
            mock_socket.send.assert_called()
            mock_socket.recv.assert_called()

    @patch(
        'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.ZmqHandshakeStrategy._zmq_ctx'
    )
    @patch('vllm.utils.make_zmq_path')
    def test_zmq_handshake_multi_rank(self, mock_make_path, mock_zmq_ctx):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock(side_effect=["agent-0", "agent-1"])

        strategy = ZmqHandshakeStrategy(mock_nixl, 1, 2, 8080, "test-engine",
                                        mock_add_agent)

        mock_socket = MagicMock()
        mock_zmq_ctx.return_value.__enter__.return_value = mock_socket
        mock_make_path.side_effect = [
            "tcp://localhost:8080", "tcp://localhost:8081"
        ]

        test_metadata = self.create_test_metadata()
        with patch('msgspec.msgpack.Decoder') as mock_decoder_class:
            mock_decoder = MagicMock()
            mock_decoder_class.return_value = mock_decoder
            mock_decoder.decode.return_value = test_metadata

            result = strategy.initiate_handshake("localhost", 8080, 2)

            assert result == {0: "agent-0", 1: "agent-1"}
            assert mock_add_agent.call_count == 2

    @patch('threading.Thread')
    def test_setup_listener(self, mock_thread):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock()

        strategy = ZmqHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                        mock_add_agent)

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        test_metadata = self.create_test_metadata()

        with patch('threading.Event') as mock_event_class:
            mock_event = MagicMock()
            mock_event_class.return_value = mock_event

            strategy.setup_listener(test_metadata)

            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            mock_event.wait.assert_called_once()

    def test_cleanup(self):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock()

        strategy = ZmqHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                        mock_add_agent)

        mock_thread = MagicMock()
        strategy._listener_thread = mock_thread

        strategy.cleanup()

        mock_thread.join.assert_called_once_with(timeout=0)


class TestHttpHandshakeStrategy:

    def create_test_metadata_response(self) -> dict:
        return {
            "0": {
                "0": {
                    "engine_id":
                    "3871ab24-6b5a-4ea5-a614-5381594bcdde",
                    "agent_metadata":
                    base64.b64encode(b"nixl-prefill-agent-data").decode(),
                    "kv_caches_base_addr": [0x7f8b2c000000],
                    "num_blocks":
                    1000,
                    "block_len":
                    128,
                    "attn_backend_name":
                    "FLASH_ATTN_VLLM_V1"
                }
            }
        }

    @patch(
        'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.urlopen')
    def test_http_handshake_success(self, mock_urlopen):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock(return_value="remote-agent-0")

        strategy = HttpHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                         mock_add_agent)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            self.create_test_metadata_response()).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = strategy.initiate_handshake("localhost", 8080, 1)

        assert result == {0: "remote-agent-0"}
        mock_add_agent.assert_called_once()

        call_args = mock_add_agent.call_args
        metadata = call_args[0][0]
        assert isinstance(metadata, NixlAgentMetadata)
        assert metadata.engine_id == "3871ab24-6b5a-4ea5-a614-5381594bcdde"
        assert metadata.agent_metadata == b"nixl-prefill-agent-data"

    @patch(
        'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.urlopen')
    def test_http_handshake_multi_rank(self, mock_urlopen):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock(return_value="remote-agent-1")

        strategy = HttpHandshakeStrategy(mock_nixl, 1, 2, 8080, "test-engine",
                                         mock_add_agent)

        response_data = {
            "0": {
                "0": {
                    "engine_id":
                    "339a1bdd-e9ad-4c6e-a3e3-e0e7cca2238d",
                    "agent_metadata":
                    base64.b64encode(b"decode-agent-0-data").decode(),
                    "kv_caches_base_addr": [0x7f8b2c000000],
                    "num_blocks":
                    800,
                    "block_len":
                    128,
                    "attn_backend_name":
                    "FLASH_ATTN_VLLM_V1"
                },
                "1": {
                    "engine_id":
                    "339a1bdd-e9ad-4c6e-a3e3-e0e7cca2238d",
                    "agent_metadata":
                    base64.b64encode(b"decode-agent-1-data").decode(),
                    "kv_caches_base_addr": [0x7f8b2d000000],
                    "num_blocks":
                    800,
                    "block_len":
                    128,
                    "attn_backend_name":
                    "FLASH_ATTN_VLLM_V1"
                }
            }
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = strategy.initiate_handshake("localhost", 8080, 2)

        assert result == {1: "remote-agent-1"}

        call_args = mock_add_agent.call_args
        metadata = call_args[0][0]
        assert metadata.agent_metadata == b"decode-agent-1-data"

    @patch(
        'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.urlopen')
    def test_http_handshake_url_error(self, mock_urlopen):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock()

        strategy = HttpHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                         mock_add_agent)

        mock_urlopen.side_effect = URLError("Connection failed")

        with pytest.raises(URLError):
            strategy.initiate_handshake("localhost", 8080, 1)

    @patch(
        'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.urlopen')
    def test_http_handshake_none_response(self, mock_urlopen):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock()

        strategy = HttpHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                         mock_add_agent)

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(None).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with pytest.raises(RuntimeError,
                           match="Remote server returned None metadata"):
            strategy.initiate_handshake("localhost", 8080, 1)

    @patch(
        'vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector.urlopen')
    def test_http_handshake_missing_rank(self, mock_urlopen):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock()

        strategy = HttpHandshakeStrategy(mock_nixl, 1, 2, 8080,
                                         "decode-engine", mock_add_agent)
        mock_response = MagicMock()
        empty_response: dict[str, dict[str, dict[str, Any]]] = {"0": {}}
        mock_response.read.return_value = json.dumps(empty_response).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with pytest.raises(RuntimeError,
                           match="No metadata found for dp_rank 0"):
            strategy.initiate_handshake("localhost", 8080, 1)

    def test_setup_listener_noop(self):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock()

        strategy = HttpHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                         mock_add_agent)

        test_metadata = NixlAgentMetadata(
            engine_id="test-engine",
            agent_metadata=b"test-data",
            kv_caches_base_addr=[12345],
            num_blocks=100,
            block_len=16,
            attn_backend_name="FLASH_ATTN_VLLM_V1")

        strategy.setup_listener(test_metadata)

    def test_cleanup_noop(self):
        mock_nixl = MagicMock()
        mock_add_agent = MagicMock()

        strategy = HttpHandshakeStrategy(mock_nixl, 0, 1, 8080, "test-engine",
                                         mock_add_agent)

        strategy.cleanup()


class TestHandshakeStrategyIntegration:

    @patch.dict('os.environ', {'VLLM_NIXL_HANDSHAKE_METHOD': 'zmq'})
    @patch('vllm.envs.VLLM_NIXL_HANDSHAKE_METHOD', 'zmq')
    def test_zmq_strategy_selection(self):
        assert envs.VLLM_NIXL_HANDSHAKE_METHOD.lower() == 'zmq'

    @patch.dict('os.environ', {'VLLM_NIXL_HANDSHAKE_METHOD': 'http'})
    @patch('vllm.envs.VLLM_NIXL_HANDSHAKE_METHOD', 'http')
    def test_http_strategy_selection(self):
        assert envs.VLLM_NIXL_HANDSHAKE_METHOD.lower() == 'http'
