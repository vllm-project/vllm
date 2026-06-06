# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared test helpers for ECCPUConnector unit tests.

Used by test_scheduler.py and test_scheduler_stress.py.
"""

import uuid
from unittest.mock import MagicMock, Mock

import pybase64
import torch

from vllm.config import VllmConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.config.model import ModelConfig
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import ECRegionLayout
from vllm.distributed.ec_transfer.ec_connector.ec_shared_region import ECSharedRegion
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

# ── layout constants ──────────────────────────────────────────────────────────

_NUM_BLOCKS = 8
_BLOCK_SIZE = 64
_HIDDEN_DIM = _BLOCK_SIZE // 2  # 32 elements × 2 bytes (fp16) = 64 bytes
_ELEMENT_SIZE = 2


# ── layout ────────────────────────────────────────────────────────────────────


def _make_layout() -> ECRegionLayout:
    """Fresh ECRegionLayout backed by a real per-test mmap file."""
    region = ECSharedRegion(
        instance_id=str(uuid.uuid4()),
        num_blocks=_NUM_BLOCKS,
        block_size_bytes=_BLOCK_SIZE,
    )
    return ECRegionLayout(
        region=region,
        dtype=torch.float16,
        hidden_dim=_HIDDEN_DIM,
        element_size=_ELEMENT_SIZE,
        block_size_bytes=_BLOCK_SIZE,
        num_blocks=_NUM_BLOCKS,
    )


# ── mock factories ────────────────────────────────────────────────────────────


def _make_nixl_mock() -> MagicMock:
    """Minimal NixlWrapper mock with deterministic return values."""
    nixl = MagicMock()
    nixl.get_agent_metadata.return_value = b"agent-meta"
    nixl.get_reg_descs.return_value = MagicMock()
    nixl.get_xfer_descs.return_value = MagicMock()
    nixl.prep_xfer_dlist.return_value = 42
    nixl.check_xfer_state.return_value = "PROC"
    nixl.add_remote_agent.return_value = "remote-agent-1"
    return nixl


def _make_vllm_config(is_producer: bool, is_consumer: bool) -> Mock:
    """Spec'd VllmConfig mock — typo'd attribute access raises AttributeError."""
    cfg = Mock(spec=VllmConfig)
    cfg.ec_transfer_config = Mock(spec=ECTransferConfig)
    cfg.ec_transfer_config.is_ec_producer = is_producer
    cfg.ec_transfer_config.is_ec_consumer = is_consumer
    cfg.ec_transfer_config.engine_id = str(uuid.uuid4())
    cfg.model_config = Mock(spec=ModelConfig)
    cfg.model_config.model = "test-model"
    return cfg


# ── request / feature helpers ─────────────────────────────────────────────────


def _feature(mm_hash: str, length: int = 1, offset: int = 0) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=mm_hash,
        mm_position=PlaceholderRange(offset=offset, length=length),
        mm_hash=mm_hash,
    )


def _request_for(
    *features: MultiModalFeatureSpec, params: dict | None = None
) -> MagicMock:
    req = MagicMock()
    req.mm_features = list(features)
    req.ec_transfer_params = params
    return req


def _info(
    *,
    peer_host: str = "host",
    peer_port: int = 1234,
    size_bytes: int = _BLOCK_SIZE,
    metadata: bytes = b"meta",
) -> dict:
    """Build the announcement-info dict the consumer expects from the producer."""
    return {
        "peer_host": peer_host,
        "peer_port": peer_port,
        "size_bytes": size_bytes,
        "nixl_agent_metadata_b64": pybase64.b64encode(metadata).decode("ascii"),
    }
