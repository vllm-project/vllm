# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden import (
    connector as connector_module,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.connector import (
    MooncakeStoreECConnector,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_hidden.data import (
    HIDDEN_TENSOR_LAYOUT,
    HiddenKeyMetadata,
    HiddenPoolKey,
    LoadSpec,
    MMMeta,
    MooncakeStoreConnectorMetadata,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange


class FakeWorker:
    def __init__(self):
        self.requests = []
        self.key_metadata = HiddenKeyMetadata(
            cache_prefix="",
            kind="encoder_output",
            model_name="qwen",
            encoder="encoder-config-a",
            storage="replicated_object",
            parallel="tp:1@pp:1@pcp:1@dcp:1@mm_tp:weights",
            tensor_layout=HIDDEN_TENSOR_LAYOUT,
        )

    def make_pool_key(self, identifier: str) -> HiddenPoolKey:
        return HiddenPoolKey(self.key_metadata, identifier)

    def enqueue_save(self, request):
        self.requests.append(request)

    def get_finished_sending(self):
        return set()

    def get_failed_sending(self):
        return {}


def make_connector(*, soft_pin_video_hidden: bool = False):
    connector = MooncakeStoreECConnector.__new__(MooncakeStoreECConnector)
    connector._is_producer = True
    connector._is_consumer = False
    connector.lookup_client = None
    connector.lookup_async = True
    connector.worker = FakeWorker()
    connector._connector_metadata = None
    connector.soft_pin_video_hidden = soft_pin_video_hidden
    connector.load_specs = {}
    connector.lookup_result_cache = {}
    connector.identifier_waiters = {}
    connector._candidate_consumes = {}
    connector._candidate_loads = {}
    connector._candidate_saves = {}
    connector._load_modalities = {}
    connector._save_modalities = {}
    return connector


class FakeLookupClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.discarded = []

    def lookup_batch(self, identifiers, non_block=True):
        self.calls.append((tuple(identifiers), non_block))
        return self.results.pop(0)

    def discard(self, identifier):
        self.discarded.append(identifier)


def make_request(request_id, features):
    mm_features = [
        MultiModalFeatureSpec(
            data=None,
            modality=modality,
            identifier=identifier,
            mm_position=PlaceholderRange(offset=offset, length=length),
        )
        for identifier, offset, length, modality in features
    ]
    return SimpleNamespace(
        request_id=request_id,
        mm_features=mm_features,
        num_tokens=1000,
    )


def make_scheduler_output(*, finished_req_ids=None, preempted_req_ids=None):
    return SimpleNamespace(
        finished_req_ids=finished_req_ids or set(),
        preempted_req_ids=preempted_req_ids,
    )


def test_build_hidden_key_metadata_uses_structured_key_fields(monkeypatch):
    monkeypatch.setattr(
        connector_module,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        connector_module,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        connector_module,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    multimodal_config = SimpleNamespace(
        compute_hash=lambda: "encoder-config-a",
        mm_encoder_tp_mode="data",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/models/qwen",
            multimodal_config=multimodal_config,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        ec_transfer_config=SimpleNamespace(
            ec_connector_extra_config={
                "cache_prefix": "shared-prefix",
                "hidden_cache_prefix": "hidden-prefix",
            }
        ),
    )

    metadata = connector_module.build_hidden_key_metadata(vllm_config)

    assert metadata.cache_prefix == "hidden-prefix"
    assert metadata.kind == "encoder_output"
    assert metadata.model_name == "qwen"
    assert metadata.encoder == "encoder-config-a"
    assert metadata.storage == "replicated_object"
    assert metadata.parallel == "tp:4@pp:2@pcp:1@dcp:1@mm_tp:data"
    assert "storage" not in metadata.parallel
    assert metadata.tensor_layout == "tensor"


def test_ensure_cache_available_defers_pending_batch_lookup():
    connector = make_connector()
    connector._is_consumer = True
    connector._is_producer = False
    connector.lookup_client = FakeLookupClient([None])
    request = make_request(
        "req-1",
        [
            ("image-1", 20, 60, "image"),
            ("image-2", 500, 60, "image"),
        ],
    )

    assert not connector.ensure_cache_available(request, num_computed_tokens=0)

    assert connector.lookup_client.calls == [
        (("image-1", "image-2"), True),
    ]
    assert connector.identifier_waiters == {
        "image-1": {"req-1"},
        "image-2": {"req-1"},
    }


def test_ensure_cache_available_deduplicates_request_waiters_and_lookup_results():
    connector = make_connector()
    connector._is_consumer = True
    connector._is_producer = False
    connector.lookup_client = FakeLookupClient(
        [
            {"image-1": True, "image-2": False},
        ]
    )
    request = make_request(
        "req-1",
        [
            ("image-1", 20, 60, "image"),
            ("image-2", 500, 60, "image"),
        ],
    )

    assert connector.ensure_cache_available(request, num_computed_tokens=0)
    assert connector.ensure_cache_available(request, num_computed_tokens=0)

    assert connector.lookup_client.calls == [
        (("image-1", "image-2"), True),
    ]
    assert connector.identifier_waiters == {
        "image-1": {"req-1"},
        "image-2": {"req-1"},
    }
    assert connector.lookup_result_cache == {
        "image-1": True,
        "image-2": False,
    }


def test_has_cache_item_is_local_only():
    connector = make_connector()
    connector._is_consumer = True
    connector._is_producer = False
    connector.lookup_client = SimpleNamespace(lookup=lambda identifier: True)
    connector.lookup_result_cache = {"image-1": True, "image-2": False}

    assert connector.has_cache_item("image-1")
    assert not connector.has_cache_item("image-2")
    assert not connector.has_cache_item("unknown")


def test_build_connector_meta_commits_waiter_consumes_and_keeps_unreached_image():
    connector = make_connector()
    connector._is_consumer = True
    connector._is_producer = False
    connector.lookup_result_cache = {"image-1": True, "image-2": True}
    connector.identifier_waiters = {
        "image-1": {"req-1"},
        "image-2": {"req-1"},
    }
    connector.load_specs["image-1"] = LoadSpec(can_load=False)
    request = make_request(
        "req-1",
        [
            ("image-1", 20, 60, "image"),
            ("image-2", 500, 60, "image"),
        ],
    )

    connector.update_state_after_alloc(request, 0)
    meta = connector.build_connector_meta(make_scheduler_output())

    assert [item.identifier for item in meta.items] == ["image-1"]
    assert "image-1" not in connector.identifier_waiters
    assert "image-1" not in connector.lookup_result_cache
    assert connector.identifier_waiters == {"image-2": {"req-1"}}
    assert connector.lookup_result_cache == {"image-2": True}


def test_build_connector_meta_rolls_back_preempted_candidate_state():
    connector = make_connector()
    connector._is_consumer = True
    connector._is_producer = False
    connector.lookup_result_cache = {"image-1": True}
    connector.identifier_waiters = {"image-1": {"req-1"}}
    connector.load_specs["image-1"] = LoadSpec(can_load=False)
    request = make_request("req-1", [("image-1", 20, 60, "image")])

    connector.update_state_after_alloc(request, 0)
    meta = connector.build_connector_meta(
        make_scheduler_output(preempted_req_ids={"req-1"})
    )

    assert meta.items == []
    assert connector.identifier_waiters == {"image-1": {"req-1"}}
    assert connector.lookup_result_cache == {"image-1": True}
    assert "image-1" in connector.load_specs


def test_build_connector_meta_cleans_finished_waiters():
    connector = make_connector()
    connector._is_consumer = True
    connector._is_producer = False
    connector.lookup_client = FakeLookupClient([])
    connector.lookup_result_cache = {"image-1": True, "image-2": True}
    connector.identifier_waiters = {
        "image-1": {"req-1"},
        "image-2": {"req-1", "req-2"},
    }

    connector.build_connector_meta(make_scheduler_output(finished_req_ids={"req-1"}))

    assert "image-1" not in connector.identifier_waiters
    assert "image-1" not in connector.lookup_result_cache
    assert connector.identifier_waiters == {"image-2": {"req-2"}}
    assert connector.lookup_result_cache == {"image-2": True}
    assert connector.lookup_client.discarded == ["image-1"]


def test_cleanup_lookup_results_discards_inflight_lookup_without_waiters():
    connector = make_connector()
    connector._is_consumer = True
    connector._is_producer = False
    connector.lookup_client = FakeLookupClient([])
    connector.identifier_waiters = {"image-1": set()}
    connector.lookup_result_cache = {"image-1": True}
    connector.load_specs["image-1"] = LoadSpec(can_load=False)

    connector._cleanup_lookup_results_without_waiters()

    assert connector.identifier_waiters == {}
    assert connector.lookup_result_cache == {}
    assert connector.load_specs == {}
    assert connector.lookup_client.discarded == ["image-1"]


def test_build_connector_meta_merges_load_and_save_item_by_identifier():
    connector = make_connector()
    connector._is_consumer = True
    connector.load_specs["video-hash"] = LoadSpec(can_load=False)
    connector.lookup_result_cache["video-hash"] = True
    connector.identifier_waiters["video-hash"] = {"req-1"}
    request = SimpleNamespace(
        request_id="req-1",
        mm_features=[
            SimpleNamespace(
                identifier="video-hash",
                modality="video",
            )
        ],
    )

    connector.update_state_after_alloc(request, 0)
    meta = connector.build_connector_meta(make_scheduler_output())

    assert len(meta.items) == 1
    item = meta.items[0]
    assert item.identifier == "video-hash"
    assert item.modality == "video"
    assert item.can_save
    assert item.load_spec is not None
    assert item.load_spec.can_load
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


def test_get_finished_logs_failed_hidden_saves(caplog):
    class FailedWorker(FakeWorker):
        def get_finished_sending(self):
            return {"image-ok"}

        def get_failed_sending(self):
            return {"image-failed": "batch put failed"}

    connector = make_connector()
    connector.worker = FailedWorker()

    finished_sending, finished_recving = connector.get_finished({"req-1"})

    assert finished_sending == {"image-ok"}
    assert finished_recving is None
    assert "hidden_store_save_failed" in caplog.text
    assert "image-failed" in caplog.text
    assert "batch put failed" in caplog.text
