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
            block_size=64, mamba_cache_mode="align", mamba_ssm_cache_dtype="float32"
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
    from dataclasses import replace

    from tests.v1.core.test_prefix_caching import (
        _make_hybrid_kv_cache_config,
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
    kv_config.kv_cache_groups[1].kv_cache_spec = replace(
        kv_config.kv_cache_groups[1].kv_cache_spec,
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
        drop_eagle_block=False,
        cache_hit_alignment_tokens=64,
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
def test_layerwise_hybrid_without_alignment_keeps_legacy_layout(hybrid_config):
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
    from vllm.v1.worker.utils import allocate_kv_cache, select_common_block_size

    config = hybrid_config
    config.cache_config.mamba_cache_mode = "none"
    config.cache_config.mamba_block_size = config.model_config.max_model_len

    Platform.update_block_size_for_backend(config)
    attention = config.compilation_config.static_forward_context[
        "model.layers.1.self_attn.attn"
    ]
    specs = [
        attention.get_kv_cache_spec(config),
        MambaSpec(
            block_size=config.cache_config.mamba_block_size,
            shapes=((1,),),
            dtypes=(torch.float32,),
            page_size_padded=config.cache_config.mamba_page_size_padded,
            mamba_cache_mode="none",
        ),
    ]
    layout = resolve_kv_cache_layout(
        config,
        [[layout.name for layout in FlashInferBackend.supported_kv_cache_layouts()]],
        specs,
    )

    assert config.cache_config.block_size > 64
    assert config.cache_config.mamba_page_size_padded is not None
    assert not layout.is_block_outermost

    kernel_block_size = select_common_block_size(
        config.cache_config.block_size, [FlashInferBackend], [specs[0]]
    )
    assert kernel_block_size < config.cache_config.block_size
    kv_config = get_kv_cache_config_from_groups(
        config,
        [
            KVCacheGroupSpec(["attention"], specs[0]),
            KVCacheGroupSpec(["mamba"], specs[1]),
        ],
        8 * sum(spec.page_size_bytes for spec in specs),
    )
    pages = allocate_kv_cache(
        kv_config,
        torch.device("cpu"),
        layout,
        [kernel_block_size, specs[1].block_size],
    )
    assert kv_config.num_blocks > 1
    assert pages["attention"].shape[1] == 2
    assert pages["attention"].shape[-2] == specs[0].get_num_kernel_states(
        kernel_block_size
    )


@pytest.mark.parametrize("layerwise", [None, False, True])
def test_offloading_layout_preserves_layerwise_hybrid_packing(hybrid_config, layerwise):
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    config = hybrid_config
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector", kv_role="kv_both"
    )
    if not layerwise:
        config.quant_config = None if layerwise is None else Fp8Config()
    required = OffloadingConnector.get_required_kvcache_layout(config)
    assert required == (None if layerwise else "LBHNC")

    Platform.update_block_size_for_backend(config)

    if layerwise:
        assert config.cache_config.block_size == 64
        assert config.cache_config.mamba_page_size_padded is None
    else:
        assert config.cache_config.block_size > 64
        assert config.cache_config.mamba_page_size_padded is not None
