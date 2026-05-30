# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import torch

from vllm.config import VllmConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.ec_connector_cache_manager import ECConnectorCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

_init = False


def _vllm_config() -> VllmConfig:
    config = Mock(spec=VllmConfig)
    config.model_config = Mock()
    config.model_config.get_hidden_size = Mock(return_value=768)
    config.model_config.dtype = torch.float16
    config.kv_transfer_config = None
    return config


def _request(req_id: str, mm_id: str, num_embeds: int = 4) -> Request:
    global _init
    if not _init:
        init_none_hash(sha256)
        _init = True
    block_hasher = get_request_block_hasher(16, sha256)
    mm_feature = MultiModalFeatureSpec(
        data=None,
        mm_position=PlaceholderRange(offset=0, length=num_embeds),
        identifier=mm_id,
        modality="image",
    )
    return Request(
        request_id=req_id,
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        mm_features=[mm_feature],
        block_hasher=block_hasher,
    )


def test_allocate_and_mark_consumer_received():
    cfg = _vllm_config()
    m = ECConnectorCacheManager(cache_size=100, vllm_config=cfg)
    req = _request("r1", "h1", num_embeds=10)
    assert m.can_allocate(req, 0, encoder_compute_budget=100, num_embeds_to_schedule=0)
    m.allocate(req, 0)
    assert m.num_free_slots == 90
    m.mark_consumer_received("h1")
    assert "h1" in m.freeable
    assert m.num_freeable_slots == 100


def test_can_allocate_evicts_freeable():
    cfg = _vllm_config()
    m = ECConnectorCacheManager(cache_size=20, vllm_config=cfg)
    r1 = _request("a", "x", num_embeds=10)
    assert m.can_allocate(r1, 0, 100, 0)
    m.allocate(r1, 0)
    m.mark_consumer_received("x")
    r2 = _request("b", "y", num_embeds=15)
    assert m.can_allocate(r2, 0, 100, 0)
    freed = m.get_freed_mm_hashes()
    assert "x" in freed


def test_free_encoder_input():
    cfg = _vllm_config()
    m = ECConnectorCacheManager(cache_size=50, vllm_config=cfg)
    req = _request("r", "z", num_embeds=5)
    m.can_allocate(req, 0, 100, 0)
    m.allocate(req, 0)
    m.free_encoder_input(req, 0)
    assert "z" in m.freeable


def test_ec_connector_capacity_embeds_default():
    from vllm.config import ECTransferConfig
    from vllm.config.ec_transfer import (
        DEFAULT_EC_CONNECTOR_CAPACITY_EMBEDS,
        EC_CONNECTOR_CAPACITY_EMBEDS_KEY,
    )

    cfg = ECTransferConfig(
        ec_connector="ECSharedMemoryConnector", ec_role="ec_producer"
    )
    assert (
        cfg.get_ec_connector_capacity_embeds() == DEFAULT_EC_CONNECTOR_CAPACITY_EMBEDS
    )

    custom = ECTransferConfig(
        ec_connector="ECSharedMemoryConnector",
        ec_role="ec_producer",
        ec_connector_extra_config={EC_CONNECTOR_CAPACITY_EMBEDS_KEY: 4096},
    )
    assert custom.get_ec_connector_capacity_embeds() == 4096


def test_cache_manager_enabled_by_connector_class():
    from vllm.config import ECTransferConfig
    from vllm.distributed.ec_transfer.ec_connector.example_connector import (
        ECExampleConnector,
    )
    from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
    from vllm.distributed.ec_transfer.ec_connector.shared_memory_connector import (
        ECSharedMemoryConnector,
    )

    assert ECExampleConnector.supports_ec_connector_cache_manager is False
    assert ECSharedMemoryConnector.supports_ec_connector_cache_manager is True

    example_cls = ECConnectorFactory.get_connector_class(
        ECTransferConfig(ec_connector="ECExampleConnector", ec_role="ec_producer")
    )
    shm_cls = ECConnectorFactory.get_connector_class(
        ECTransferConfig(ec_connector="ECSharedMemoryConnector", ec_role="ec_producer")
    )
    assert example_cls.supports_ec_connector_cache_manager is False
    assert shm_cls.supports_ec_connector_cache_manager is True
