# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.model_loader.reload.inplace as reload_inplace_module
from vllm.lora.layers import LoRAMappingType
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheSpec, get_block_table_width
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.v1.worker.utils import select_common_block_size


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """Several tests here set the process-wide default dtype to float16 and
    previously leaked it, corrupting later float-sensitive tests in the same
    pytest process (torch.randn silently produced fp16)."""
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)


def _make_mock_backend_for_kernel_block_size(
    supported_sizes: list[int | MultipleOf],
):
    class _MockBackend:
        @staticmethod
        def get_name():
            return "MOCK"

        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return supported_sizes

    return _MockBackend()


def test_select_common_block_size_prefers_manager_block_size():
    backend_a = _make_mock_backend_for_kernel_block_size([MultipleOf(32)])
    backend_b = _make_mock_backend_for_kernel_block_size([64, MultipleOf(16)])

    selected_size = select_common_block_size(128, [backend_a, backend_b])
    assert selected_size == 128


def test_select_common_block_size_uses_largest_shared_int():
    backend_a = _make_mock_backend_for_kernel_block_size([128, 64])
    backend_b = _make_mock_backend_for_kernel_block_size([64, 32])

    selected_size = select_common_block_size(256, [backend_a, backend_b])
    assert selected_size == 64


def test_select_common_block_size_without_active_backends_uses_manager_size():
    assert select_common_block_size(256, []) == 256


def test_select_common_block_size_accepts_rocm_sparse_block_size_16(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)

    selected_size = select_common_block_size(
        16,
        [DeepseekV32IndexerBackend, ROCMAiterMLASparseBackend],  # type: ignore[type-abstract]  # not instantiated
    )
    assert selected_size == 16


def test_select_common_block_size_no_valid_option():
    backend_a = _make_mock_backend_for_kernel_block_size([64])
    backend_b = _make_mock_backend_for_kernel_block_size([MultipleOf(16)])

    with pytest.raises(ValueError):
        select_common_block_size(48, [backend_a, backend_b])


def _mock_backend(supported: list, *, exact: bool = False):
    """Backend accepting multiples of `supported`, or only those exact sizes
    when `exact` (as CPU_MLA does)."""
    from vllm.v1.attention.backend import AttentionBackend

    class _MockBackendCls(AttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "MOCK_EXACT" if exact else "MOCK"

        @staticmethod
        def get_supported_kernel_block_sizes(
            kv_cache_spec: KVCacheSpec | None = None,
        ) -> list[int | MultipleOf]:
            return list(supported)

        if exact:

            @classmethod
            def supports_block_size(cls, block_size: int | None) -> bool:
                return block_size is None or block_size in supported

    return _MockBackendCls


@pytest.mark.parametrize(
    "backends,expected",
    [
        # A lone backend keeps the default; MultipleOf(8) makes a
        # minimum-based regression return 8 instead of 16.
        ([[MultipleOf(8)]], 16),
        # Sparse-MLA main backend (32 or 64) beside an indexer supporting only
        # 64: the main backend alone picks 32, which the indexer rejects at
        # select_common_block_size time.
        ([[32, 64], [64]], 64),
        ([[32, 64]], 32),
        # Neither divides the other, so only the LCM satisfies both.
        ([[32], [48]], 96),
    ],
)
def test_preferred_block_size_satisfies_every_backend(backends, expected):
    classes = [_mock_backend(s) for s in backends]
    assert Platform._preferred_block_size_for_backends(classes, 16, None) == expected


def test_preferred_block_size_searches_past_an_exact_size_backend():
    # Extending greedily picks lcm(16, 32) = 32, which the exact backend
    # rejects; 96 is accepted by both.
    classes = [_mock_backend([16, 96], exact=True), _mock_backend([MultipleOf(32)])]
    assert Platform._preferred_block_size_for_backends(classes, 16, None) == 96


def test_preferred_block_size_rejects_backends_with_no_common_size():
    classes = [_mock_backend([16], exact=True), _mock_backend([MultipleOf(64)])]
    with pytest.raises(ValueError, match="share no supported KV cache block size"):
        Platform._preferred_block_size_for_backends(classes, 16, None)


@pytest.mark.parametrize("cpu", [False, True])
def test_alignment_rejected_by_a_sibling_backend_raises(monkeypatch, cpu):
    # Alignment that lands on a size an exact-size sibling rejects must fail
    # here, not later in select_common_block_size().
    backends = [_mock_backend([MultipleOf(16)]), _mock_backend([16], exact=True)]
    monkeypatch.setattr(
        Platform, "_find_non_ssm_backends", classmethod(lambda cls, c: backends)
    )

    def bump(cls, vllm_config, backend_cls):
        vllm_config.cache_config.block_size = 32

    monkeypatch.setattr(Platform, "_align_hybrid_block_size", classmethod(bump))
    cache_config = SimpleNamespace(
        block_size=16, user_specified_block_size=True, kv_cache_dtype_skip_layers=None
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config, model_config=SimpleNamespace(is_hybrid=True)
    )
    from vllm.platforms.cpu import CpuPlatform  # overrides the method; must check too

    platform = CpuPlatform if cpu else Platform
    with pytest.raises(ValueError, match="MOCK_EXACT"):
        platform.update_block_size_for_backend(vllm_config)


class _CheckpointingMambaLayer(MambaBase):
    mamba_type = MambaAttentionBackendEnum.GDN_ATTN

    def __init__(self, prefill_checkpoint_alignment: int | None):
        self._prefill_checkpoint_alignment = prefill_checkpoint_alignment

    def get_prefill_checkpoint_alignment(self) -> int | None:
        return self._prefill_checkpoint_alignment

    def get_state_shape(self):
        return ()

    def get_state_dtype(self):
        return ()


def _align_qwen3_5_blocks(*prefill_checkpoint_alignments: int | None):
    from tests.v1.attention.utils import create_vllm_config

    vllm_config = create_vllm_config(model_name="Qwen/Qwen3.5-0.8B")
    cache_config = vllm_config.cache_config
    cache_config.mamba_cache_mode = "align"
    forward_context = vllm_config.compilation_config.static_forward_context
    for i, alignment in enumerate(prefill_checkpoint_alignments):
        forward_context[f"mamba.{i}"] = _CheckpointingMambaLayer(alignment)
    Platform._align_hybrid_block_size(vllm_config, _mock_backend([MultipleOf(16)]))
    return cache_config


@pytest.mark.parametrize(
    ("prefill_checkpoint_alignments", "block_size"),
    [((64,), 576), ((None,), 544), ((64, None), 544)],
)
def test_blocks_land_on_the_mamba_checkpoint_grid(
    prefill_checkpoint_alignments, block_size
):
    """A layer that checkpoints every 64 tokens needs 64-aligned blocks.

    Unaligned, Qwen3.5-0.8B's block (544) puts every other checkpoint off the
    grid. A Mamba layer that cannot checkpoint (like Qwen4Exp's PLE short conv)
    stops the scheduler from taking any, so the block stays.
    """
    cache_config = _align_qwen3_5_blocks(*prefill_checkpoint_alignments)

    assert cache_config.block_size == block_size


def test_xpu_gdn_rounding_rejected_by_a_sibling_backend_raises(monkeypatch):
    # XPU's GDN rounding runs after super()'s check, so it must check again.
    try:
        from vllm.platforms.xpu import XPUPlatform
    except ImportError:
        pytest.skip("vllm_xpu_kernels not importable outside an XPU stack")

    backends = [_mock_backend([MultipleOf(16)]), _mock_backend([16], exact=True)]
    monkeypatch.setattr(
        Platform, "update_block_size_for_backend", classmethod(lambda cls, c: None)
    )
    monkeypatch.setattr(
        Platform, "_find_non_ssm_backends", classmethod(lambda cls, c: backends)
    )
    gdn_layer = SimpleNamespace(
        get_attn_backend=lambda: SimpleNamespace(get_name=lambda: "GDN_ATTN")
    )
    monkeypatch.setattr(
        "vllm.config.vllm.get_layers_from_vllm_config",
        lambda vllm_config, layer_type: {"gdn": gdn_layer},
    )
    cache_config = SimpleNamespace(
        block_size=16,
        mamba_cache_mode="none",
        mamba_page_size_padded=None,
        user_specified_block_size=True,
        kv_cache_dtype_skip_layers=None,
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config, model_config=SimpleNamespace(is_hybrid=True)
    )
    with pytest.raises(ValueError, match="MOCK_EXACT"):
        XPUPlatform.update_block_size_for_backend(vllm_config)


def test_set_active_mm_loras_builds_tower_and_connector_mappings():
    model = Mock()
    model.get_mm_lora_token_counts.side_effect = (
        lambda *, modality, mm_kwargs, num_mm_embeds: (
            num_mm_embeds + 1,
            num_mm_embeds + 11,
        )
    )
    model.get_mm_mapping.return_value = SimpleNamespace(connector=True)

    lora_manager = Mock()
    lora_manager.supports_tower_connector_lora.return_value = True

    encoder_cache = EncoderCache()
    encoder_cache.mm_features["req-with-lora"] = [
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="img-0",
            mm_position=PlaceholderRange(offset=0, length=2),
        ),
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="img-1",
            mm_position=PlaceholderRange(offset=2, length=3),
        ),
    ]
    encoder_cache.mm_features["req-no-lora"] = [
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier="img-2",
            mm_position=PlaceholderRange(offset=0, length=1),
        )
    ]

    lora_state = LoraState(max_num_reqs=4)
    lora_request = LoRARequest("vision-lora", 7, "/tmp/vision-lora")
    lora_state.add_request("req-with-lora", 0, lora_request)
    lora_state.add_request("req-no-lora", 1, None)

    set_active_mm_loras(
        model=model,
        lora_manager=lora_manager,
        encoder_cache=encoder_cache,
        req_id_to_index={
            "req-with-lora": 0,
            "req-no-lora": 1,
        },
        lora_state=lora_state,
        scheduled_encoder_inputs={
            "req-with-lora": [1, 0],
            "req-no-lora": [0],
            "missing-req": [0],
        },
    )

    assert lora_manager.set_active_adapters.call_count == 2

    tower_requests, tower_mapping = lora_manager.set_active_adapters.call_args_list[
        0
    ].args
    assert tower_requests == {lora_request}
    assert tower_mapping.type is LoRAMappingType.TOWER
    assert tower_mapping.prompt_mapping == (7, 7, 0)
    assert tower_mapping.index_mapping == (7, 7, 7, 7, 7, 7, 7, 0, 0)

    connector_requests, connector_mapping = (
        lora_manager.set_active_adapters.call_args_list[1].args
    )
    assert connector_requests == {lora_request}
    assert connector_mapping.type is LoRAMappingType.CONNECTOR
    assert connector_mapping.prompt_mapping == (7, 7, 0)
    assert connector_mapping.index_mapping == ((7,) * 14 + (7,) * 13 + (0,) * 12)


def test_set_active_mm_loras_follows_modality_sorted_encoder_order():
    model = Mock()
    model.get_mm_lora_token_counts.side_effect = (
        lambda *, modality, mm_kwargs, num_mm_embeds: (num_mm_embeds, None)
    )
    lora_manager = Mock()
    lora_manager.supports_tower_connector_lora.return_value = True

    encoder_cache = EncoderCache()
    for req_id, modality, length in [("req-a", "video", 1), ("req-b", "image", 2)]:
        encoder_cache.mm_features[req_id] = [
            MultiModalFeatureSpec(
                data=None,
                modality=modality,
                identifier=f"{req_id}-0",
                mm_position=PlaceholderRange(offset=0, length=length),
            )
        ]
    lora_state = LoraState(max_num_reqs=2)
    lora_request = LoRARequest("vision-lora", 7, "/tmp/vision-lora")
    lora_state.add_request("req-a", 0, lora_request)
    lora_state.add_request("req-b", 1, None)

    set_active_mm_loras(
        model=model,
        lora_manager=lora_manager,
        encoder_cache=encoder_cache,
        req_id_to_index={"req-a": 0, "req-b": 1},
        lora_state=lora_state,
        scheduled_encoder_inputs={"req-a": [0], "req-b": [0]},
    )

    # The image is encoded before the video, so its tokens come first.
    _, tower_mapping = lora_manager.set_active_adapters.call_args_list[0].args
    assert tower_mapping.index_mapping == (0, 0, 7)


def test_reload_weights_path_replaces_object_storage_source(monkeypatch):
    # An engine started from an object-storage URI keeps that URI in
    # model_config.model_weights, which the runai_streamer loader prefers over
    # model_config.model. A new weights_path must replace it, or the loader
    # silently re-streams the original checkpoint.
    loader = Mock()
    loader.reload_weights_inplace.return_value = False
    loader.get_all_weights.return_value = iter(())
    monkeypatch.setattr(reload_inplace_module, "get_model_loader", lambda _: loader)
    monkeypatch.setattr(reload_inplace_module, "start_reload", Mock())
    monkeypatch.setattr(reload_inplace_module, "finish_reload", Mock())

    vllm_config = SimpleNamespace(
        lora_config=None,
        load_config=SimpleNamespace(load_format="runai_streamer", device=None),
        device_config=SimpleNamespace(device="cpu"),
        model_config=SimpleNamespace(
            model="/tmp/pulled-config-files",
            model_weights="s3://bucket/original",
            revision="abc123",
            quantization=None,
        ),
    )
    model = Mock()
    model.named_parameters.return_value = []
    model.load_weights.return_value = None

    reload_inplace_module.reload_weights(
        vllm_config,
        model,
        weights_iterator=None,
        weights_path="org/new-model",
        is_checkpoint_format=True,
    )

    loader.get_all_weights.assert_called_once_with(vllm_config.model_config, model)
    cfg = vllm_config.model_config
    assert (cfg.model, cfg.model_weights, cfg.revision) == ("org/new-model", "", None)


def test_init_kv_cache_with_kv_sharing_invalid_target_layer_order(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} must come before the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
                kv_sharing_target_layer_name=layer_1,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_layer_not_exist(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    invalid_layer = "model.layers.0.cross_attn.attn"
    error_msg = f"{invalid_layer} is not a valid Attention layer in the model"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                # invalid layer: cross_attn.atn doesn't exist!
                kv_sharing_target_layer_name=invalid_layer,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_same_as_current(default_vllm_config):
    torch.set_default_dtype(torch.float16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} cannot be the same as the current layer"
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name=layer_1,
            ),
        }
        # suppress var not used error
        assert fwd_context is not None


def test_v2_runner_snapshots_late_interleave_adjustment(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as v2_model_runner_module

    runner = object.__new__(v2_model_runner_module.GPUModelRunner)
    runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=16)
    runner.cp_interleave = 1

    class StopInitialization(Exception):
        pass

    monkeypatch.setattr(
        v2_model_runner_module,
        "deepcopy",
        Mock(side_effect=StopInitialization),
    )

    with pytest.raises(StopInitialization):
        runner.initialize_kv_cache(SimpleNamespace())

    assert runner.cp_interleave == 16


def test_get_block_table_width_aligns_to_128_tokens():
    assert get_block_table_width(1875, 64) == 1876
