# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.connector import (
    MooncakeStoreECConnector,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_LAYOUT_VERSION,
    HiddenKeyMetadata,
    HiddenPoolKey,
    LoadSpec,
    MMMeta,
    MooncakeStoreConnectorMetadata,
)


class FakeWorker:
    def __init__(self):
        self.requests = []
        self.key_metadata = HiddenKeyMetadata(
            model_name="qwen",
            mm_encoder_config_hash="encoder-config-a",
            hidden_parallel_key=(
                "tp:1@pp:1@pcp:1@dcp:1@mm_tp:weights@storage:replicated"
            ),
            layout=HIDDEN_LAYOUT_VERSION,
        )

    def make_pool_key(self, identifier: str) -> HiddenPoolKey:
        return HiddenPoolKey(self.key_metadata, identifier)

    def enqueue_save(self, request):
        self.requests.append(request)


def make_connector(*, soft_pin_video_hidden: bool = False):
    connector = MooncakeStoreECConnector.__new__(MooncakeStoreECConnector)
    connector._is_producer = True
    connector._is_consumer = False
    connector.worker = FakeWorker()
    connector._connector_metadata = None
    connector.soft_pin_video_hidden = soft_pin_video_hidden
    connector.load_specs = {}
    connector._load_identifiers_to_schedule = set()
    connector._save_identifiers_to_schedule = set()
    connector._load_modalities = {}
    connector._save_modalities = {}
    return connector


def test_build_connector_meta_merges_load_and_save_item_by_identifier():
    connector = make_connector()
    connector._is_consumer = True
    connector.load_specs["video-hash"] = LoadSpec(can_load=False)
    request = SimpleNamespace(
        mm_features=[
            SimpleNamespace(
                identifier="video-hash",
                modality="video",
            )
        ],
    )

    connector.update_state_after_alloc(request, 0)
    meta = connector.build_connector_meta(SimpleNamespace(finished_req_ids=set()))

    assert len(meta.items) == 1
    item = meta.items[0]
    assert item.identifier == "video-hash"
    assert item.modality == "video"
    assert item.can_save
    assert item.load_spec is not None
    assert item.load_spec.can_load
    assert meta.unfinished_identifiers == {"video-hash"}
    assert connector.load_specs == {}


def test_save_caches_skips_items_without_save_plan():
    connector = make_connector()
    connector.bind_connector_metadata(
        MooncakeStoreConnectorMetadata(
            items=[
                MMMeta(
                    identifier="image-hash",
                    modality="image",
                    can_save=False,
                )
            ]
        )
    )

    connector.save_caches({"image-hash": torch.zeros((1, 2))}, "image-hash")

    assert connector.worker.requests == []


def test_save_caches_enqueues_video_hidden_with_soft_pin():
    connector = make_connector(soft_pin_video_hidden=True)
    tensor = torch.zeros((1, 2))
    connector.bind_connector_metadata(
        MooncakeStoreConnectorMetadata(
            items=[
                MMMeta(
                    identifier="video-hash",
                    modality="video",
                    can_save=True,
                    load_spec=LoadSpec(can_load=False),
                )
            ]
        )
    )

    connector.save_caches({"video-hash": tensor}, "video-hash")

    assert len(connector.worker.requests) == 1
    request = connector.worker.requests[0]
    assert request.identifier == "video-hash"
    assert request.tensor is tensor
    assert request.with_soft_pin


def test_save_caches_does_not_soft_pin_image_hidden():
    connector = make_connector(soft_pin_video_hidden=True)
    connector.bind_connector_metadata(
        MooncakeStoreConnectorMetadata(
            items=[
                MMMeta(
                    identifier="image-hash",
                    modality="image",
                    can_save=True,
                )
            ]
        )
    )

    connector.save_caches({"image-hash": torch.zeros((1, 2))}, "image-hash")

    assert len(connector.worker.requests) == 1
    assert not connector.worker.requests[0].with_soft_pin
