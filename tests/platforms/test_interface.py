# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import Qwen3NextConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.abstract import (
    MambaBase,
    get_mamba_prefill_checkpoint_grid,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.platforms import current_platform
from vllm.platforms.interface import Platform
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="Requires FlashInfer CUDA"
    ),
]


class _LayerwiseKVConfig(Fp8Config):
    def has_layerwise_kv_cache(self):
        return True


@pytest.fixture
def hybrid_config(tmp_path, dist_init, monkeypatch, request):
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    num_heads = getattr(request, "param", 1)
    Qwen3NextConfig(
        architectures=["Qwen3NextForCausalLM"],
        hidden_size=128 * num_heads,
        head_dim=128,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        cache_config=CacheConfig(
            block_size=64,
            enable_prefix_caching=True,
            mamba_ssm_cache_dtype="float32",
        ),
        quant_config=_LayerwiseKVConfig(),
    )
    config.model_config = ModelConfig(
        model=str(tmp_path),
        dtype="bfloat16",
        max_model_len=128,
        skip_tokenizer_init=True,
    )
    with set_current_vllm_config(config), torch.device("cuda"):
        Attention(
            num_heads=num_heads,
            head_size=128,
            scale=128**-0.5,
            cache_config=config.cache_config,
            attn_backend=FlashInferBackend,
            prefix="model.layers.1.self_attn.attn",
        )
    return config


@pytest.mark.parametrize("skip_layers", [[], ["0"]])
@pytest.mark.parametrize("layout", [None, "BLHNC"])
def test_layerwise_hybrid_keeps_backend_block_size(hybrid_config, skip_layers, layout):
    config = hybrid_config
    config.cache_config.kv_cache_dtype_skip_layers = skip_layers
    config.cache_config.kv_cache_layout = layout

    Platform.update_block_size_for_backend(config)

    assert config.cache_config.block_size == 64
    assert config.cache_config.mamba_block_size == 64
    assert config.cache_config.mamba_page_size_padded is None


@pytest.mark.parametrize("layerwise,layout", [(True, "LBHNC"), (False, None)])
def test_layerwise_hybrid_fallback_keeps_legacy_alignment(
    hybrid_config, layerwise, layout
):
    config = hybrid_config
    config.cache_config.kv_cache_layout = layout
    if not layerwise:
        config.quant_config = Fp8Config()

    Platform.update_block_size_for_backend(config)

    assert config.cache_config.block_size > 64
    assert config.cache_config.mamba_page_size_padded is not None


@pytest.mark.parametrize("hybrid_config", [1, 2], indirect=True)
def test_layerwise_hybrid_rejects_unsupported_manager_block_size(hybrid_config):
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.worker.utils import select_common_block_size

    hybrid_config.cache_config.block_size = 48
    # Divisibility accepts 48, but the worker would select 16-token pages.
    assert FlashInferBackend.supports_block_size(48)
    assert select_common_block_size(48, [FlashInferBackend]) == 16

    with pytest.raises(
        ValueError, match="block size 48.*FLASHINFER.*Omit --block-size"
    ):
        Platform.update_block_size_for_backend(hybrid_config)


@pytest.mark.parametrize("block_size", [16, 32])
def test_packed_hybrid_checkpoint_is_exportable_and_reusable(hybrid_config, block_size):
    from tests.v1.core.test_prefix_caching import (
        _make_hybrid_kv_cache_config,
        _replace_group,
        make_kv_cache_manager,
        make_request,
    )
    from vllm.model_executor.layers.mamba.checkpoint import (
        compute_mamba_prefill_checkpoints,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash

    class CheckpointingGDNLayer(MambaBase):
        mamba_type = MambaAttentionBackendEnum.GDN_ATTN

        def get_prefill_checkpoint_alignment(self):
            return 64

        def get_state_shape(self):
            return ()

        def get_state_dtype(self):
            return ()

    config = hybrid_config
    config.cache_config.block_size = block_size
    config.compilation_config.static_forward_context["mamba"] = CheckpointingGDNLayer()
    Platform.update_block_size_for_backend(config)
    alignment = get_mamba_prefill_checkpoint_grid(config)
    assert alignment == config.cache_config.block_size == 64
    assert config.cache_config.mamba_page_size_padded is None

    kv_config = _make_hybrid_kv_cache_config(64, 100, ["full", "mamba"])
    _replace_group(
        kv_config,
        1,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=alignment,
    )
    init_none_hash(sha256)
    manager = make_kv_cache_manager(
        kv_config, max_model_len=128, enable_caching=True, hash_block_size=64
    )
    tokens = list(range(100))
    request = make_request("producer", tokens, 64, sha256)
    assert manager.allocate_slots(request, len(tokens)) is not None
    offsets, cols = compute_mamba_prefill_checkpoints(
        [100],
        [100],
        hash_block_size=64,
        mamba_block_size=config.cache_config.mamba_block_size,
        checkpoint_alignment=alignment,
        cache_hit_alignment_tokens=64,
        num_hash_lookahead_tokens=0,
    )
    assert offsets == [64]
    checkpoint = manager.block_pool.get_cached_block(request.block_hashes[0], [1])
    assert checkpoint is not None
    assert checkpoint[0].block_hash_num_tokens == 64
    assert (
        checkpoint[0].block_id == manager.get_block_ids(request.request_id)[1][cols[0]]
    )

    request.num_computed_tokens = len(tokens)
    manager.free(request)
    manager.new_step_starts()
    replay = make_request("replay", tokens, 64, sha256)
    _, hit_tokens, _ = manager.get_computed_blocks(replay)
    assert hit_tokens == 64


@pytest.mark.parametrize("hybrid_config", [2], indirect=True)
def test_layerwise_hybrid_without_prefix_caching_keeps_legacy_layout(hybrid_config):
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.kv_cache_interface import KVCacheGroup, MambaSpec
    from vllm.v1.worker.utils import allocate_kv_cache, select_common_block_size

    config = hybrid_config
    config.cache_config.enable_prefix_caching = False
    config.cache_config.mamba_block_size = config.model_config.max_model_len

    Platform.update_block_size_for_backend(config)
    attention = config.compilation_config.static_forward_context[
        "model.layers.1.self_attn.attn"
    ]
    attn_layer = attention.get_kv_cache_config(config)
    assert attn_layer is not None
    mamba_spec = MambaSpec(
        block_size=config.cache_config.mamba_block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        page_size_padded=config.cache_config.mamba_page_size_padded,
    )
    mamba_layer = MambaManager.layer_config(mamba_spec)
    layout = resolve_kv_cache_layout(
        config,
        [[layout.name for layout in FlashInferBackend.supported_kv_cache_layouts()]],
        [attn_layer, mamba_layer],
    )

    assert config.cache_config.block_size > 64
    assert config.cache_config.mamba_page_size_padded is not None
    assert not layout.is_block_outermost

    kernel_block_size = select_common_block_size(
        config.cache_config.block_size, [FlashInferBackend], [attn_layer.spec]
    )
    assert kernel_block_size < config.cache_config.block_size
    kv_config = get_kv_cache_config_from_groups(
        config,
        [
            KVCacheGroup.from_layers({"attention": attn_layer}),
            KVCacheGroup.from_layers({"mamba": mamba_layer}),
        ],
        8 * (attn_layer.spec.page_size_bytes + mamba_spec.page_size_bytes),
    )
    pages = allocate_kv_cache(
        kv_config,
        torch.device("cpu"),
        layout,
        [kernel_block_size, mamba_spec.block_size],
    )
    assert kv_config.num_blocks > 1
    assert pages["attention"].shape[1] == 2
    assert pages["attention"].shape[-2] == attn_layer.spec.get_num_kernel_states(
        kernel_block_size
    )
