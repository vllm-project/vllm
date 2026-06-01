# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)


class TestP2pNcclConnectorRequestIdParsing(unittest.TestCase):
    def test_extracts_prefill_and_decode_addresses(self):
        request_id = (
            "___prefill_addr_10.0.0.1:14579"
            "___decode_addr_10.0.0.2:14580_deadbeefdeadbeefdeadbeefdeadbeef"
        )

        self.assertEqual(
            P2pNcclConnector.parse_request_id(request_id, True),
            ("10.0.0.2", 14580),
        )
        self.assertEqual(
            P2pNcclConnector.parse_request_id(request_id, False),
            ("10.0.0.1", 14579),
        )

    def test_rejects_plain_offline_request_id(self):
        with self.assertRaisesRegex(ValueError, "does not contain hostname and port"):
            P2pNcclConnector.parse_request_id("0-9000a856", True)
