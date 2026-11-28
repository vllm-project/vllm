# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDP replay page (`--use-replayssm` on GDP layers) matches FlashInfer's
`gdp_conv_replay` contract: five states per block in the op's shapes and
dtypes, each starting 16-byte aligned in the packed page."""

from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.abstract import get_mamba_num_speculative_blocks
from vllm.models.nemotron_shortcut.nvidia import gdp as gdp_module
from vllm.models.nemotron_shortcut.nvidia import gdp_replay as gdp_replay_module
from vllm.models.nemotron_shortcut.nvidia.gdp import (
    NemotronShortcutGDPMixer,
    gdp_replay_conv_rows,
    gdp_replay_ring_slots,
    gdp_replay_state_dtype,
    gdp_replay_state_shape,
)
from vllm.models.nemotron_shortcut.nvidia.gdp_replay import validate_gdp_replay

from .utils import make_config

# The geometry the op supports: K / Q 128, V 64, 3 householder steps, width 4.
KEY_DIM, VALUE_DIM, HOUSEHOLDER, CONV_KERNEL = 128, 64, 3, 4


def _replay_shape(heads: int, tp_size: int, num_spec: int):
    return gdp_replay_state_shape(
        tp_size, heads, heads, KEY_DIM, VALUE_DIM, CONV_KERNEL, HOUSEHOLDER, num_spec
    )


@pytest.mark.parametrize("num_spec", [0, 1, 3, 7])
@pytest.mark.parametrize("heads,tp_size", [(16, 1), (16, 8), (8, 2)])
def test_replay_page_has_the_op_shapes(heads, tp_size, num_spec):
    tokens = 1 + num_spec
    local = heads // tp_size
    conv_channels = local * (HOUSEHOLDER * (VALUE_DIM + KEY_DIM) + KEY_DIM)
    ring = gdp_replay_ring_slots(tokens)

    assert _replay_shape(heads, tp_size, num_spec) == (
        (gdp_replay_conv_rows(tokens), conv_channels),
        (local, VALUE_DIM, KEY_DIM),
        (local, ring, HOUSEHOLDER, KEY_DIM),
        (local, ring, HOUSEHOLDER, VALUE_DIM),
        (local, ring),
    )
    assert conv_channels == 704 * local


@pytest.mark.parametrize("tokens", range(1, 9))
def test_ring_sizes_match_flashinfer(tokens):
    replay = pytest.importorskip("flashinfer.gdp_conv_replay_decode")
    assert gdp_replay_ring_slots(tokens) == replay.ring_slots(tokens, "banked")
    assert gdp_replay_conv_rows(tokens) == replay.conv_state_rows(tokens)


@pytest.mark.parametrize("tokens", [0, 9])
def test_replay_rejects_more_than_eight_tokens(tokens):
    with pytest.raises(ValueError, match="1..8 tokens"):
        gdp_replay_ring_slots(tokens)


@pytest.mark.parametrize("ssm_dtype", ["auto", "float16", "bfloat16", "float32"])
@pytest.mark.parametrize("num_spec", [0, 3, 7])
@pytest.mark.parametrize("local_heads", [2, 4, 10])
def test_replay_states_are_16_byte_aligned(local_heads, num_spec, ssm_dtype):
    """`bind_kv_cache` packs the states back to back, and the op needs every
    state, and the block stride, 16-byte aligned."""
    shapes = _replay_shape(local_heads, 1, num_spec)
    dtypes = gdp_replay_state_dtype(torch.bfloat16, "auto", ssm_dtype)
    offset = 0
    for shape, dtype in zip(shapes, dtypes, strict=True):
        assert offset % 16 == 0
        offset += prod(shape) * dtype.itemsize
    assert offset % 16 == 0


@pytest.mark.parametrize(
    "ssm_dtype,checkpoint",
    [
        ("auto", torch.float16),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
    ],
)
def test_replay_page_dtypes(ssm_dtype, checkpoint):
    assert gdp_replay_state_dtype(torch.bfloat16, "auto", ssm_dtype) == (
        torch.bfloat16,
        checkpoint,
        torch.float16,
        torch.float16,
        torch.float32,
    )


def test_replay_rejects_fp8_checkpoint():
    with pytest.raises(ValueError, match="GDP replay checkpoint"):
        gdp_replay_state_dtype(torch.bfloat16, "auto", "fp8_e4m3")


def test_replay_rejects_dim_first_conv_state(monkeypatch):
    monkeypatch.setattr(gdp_module, "is_conv_state_dim_first", lambda: True)
    with pytest.raises(ValueError, match="token-major"):
        _replay_shape(8, 1, 0)


def test_the_mixer_selects_the_page_by_the_flag(dist_init, monkeypatch):
    from vllm.config import get_current_vllm_config
    from vllm.forward_context import set_forward_context

    monkeypatch.setattr(gdp_module, "missing_gdp_kernels", lambda: ())
    monkeypatch.setattr(gdp_module, "gdp_replay_kernel_missing", lambda: False)
    config = make_config(
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_head_dim=VALUE_DIM,
        gdp_key_head_dim=KEY_DIM,
        gdp_num_householder=HOUSEHOLDER,
        gdp_conv_kernel=CONV_KERNEL,
    )
    with set_forward_context(None, get_current_vllm_config()):
        plain = NemotronShortcutGDPMixer(config, cache_config=CacheConfig(), prefix="p")
        replay = NemotronShortcutGDPMixer(
            config, cache_config=CacheConfig(use_replayssm=True), prefix="r"
        )
    plain.num_spec = replay.num_spec = 3

    assert len(plain.get_state_shape()) == 2
    assert replay.get_state_shape() == _replay_shape(8, 1, 3)


@pytest.mark.parametrize("use_replayssm,blocks", [(False, 3), (True, 0)])
def test_replay_reserves_no_per_draft_state_blocks(use_replayssm, blocks):
    """Replay verifies the whole window off one checkpoint."""
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(use_replayssm=use_replayssm),
        num_speculative_tokens=3,
    )
    assert get_mamba_num_speculative_blocks(vllm_config) == blocks


def _vllm_config(
    heads=16,
    tp_size=8,
    num_spec=3,
    prefix_caching=False,
    key_dim=KEY_DIM,
    kv_transfer=False,
    dtype=torch.bfloat16,
    conv_cache_dtype="auto",
    pp_size=1,
    ubatching=False,
    adaptive_verification=False,
) -> SimpleNamespace:
    """The fields `validate_gdp_replay` reads."""
    text_config = SimpleNamespace(
        gdp_num_heads=heads,
        gdp_key_head_dim=key_dim,
        gdp_head_dim=VALUE_DIM,
        gdp_num_householder=HOUSEHOLDER,
        gdp_conv_kernel=CONV_KERNEL,
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            use_replayssm=True,
            enable_prefix_caching=prefix_caching,
            mamba_cache_dtype=conv_cache_dtype,
        ),
        model_config=SimpleNamespace(
            hf_text_config=text_config, architecture="X", dtype=dtype
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            use_ubatching=ubatching,
        ),
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=adaptive_verification
        ),
        num_speculative_tokens=num_spec,
        kv_transfer_config=(
            SimpleNamespace(is_kv_transfer_instance=True) if kv_transfer else None
        ),
    )


@pytest.fixture
def sm100(monkeypatch):
    monkeypatch.setattr(
        gdp_replay_module.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 100,
    )


def test_validator_accepts_a_supported_config(sm100):
    config = _vllm_config()
    validate_gdp_replay(config)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"key_dim": 64}, "key head dim 128"),
        ({"tp_size": 16}, "even number of GDP heads per TP rank"),
        ({"num_spec": 8}, "1 to 7 speculative tokens"),
        ({"num_spec": 0}, "needs speculative decoding"),
        ({"prefix_caching": True}, "does not support prefix caching"),
        ({"dtype": torch.float16}, "bfloat16 model and conv cache"),
        ({"conv_cache_dtype": "float32"}, "bfloat16 model and conv cache"),
        ({"pp_size": 2}, "pipeline_parallel_size=1"),
        ({"ubatching": True}, "microbatching"),
        ({"adaptive_verification": True}, "adaptive verification"),
        ({"kv_transfer": True}, "does not support KV connectors"),
    ],
)
def test_validator_rejects_what_the_op_cannot_run(sm100, overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_gdp_replay(_vllm_config(**overrides))


def test_validator_rejects_other_gpus(monkeypatch):
    monkeypatch.setattr(
        gdp_replay_module.current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    with pytest.raises(ValueError, match="SM100-class"):
        validate_gdp_replay(_vllm_config())


@pytest.mark.parametrize(
    "use_replayssm,heads,expected",
    [(True, 16, True), (True, None, False), (False, 16, False)],
)
def test_replayssm_selects_gdp_replay_only_on_gdp_models(
    use_replayssm, heads, expected
):
    """`--use-replayssm` selects GDP replay on GDP models only."""
    config = _vllm_config(heads=heads)
    config.cache_config.use_replayssm = use_replayssm
    assert VllmConfig.use_gdp_replayssm.fget(config) is expected


@pytest.mark.parametrize("heads,expected", [(16, False), (None, True)])
def test_gdp_replay_stays_off_the_mamba2_flashinfer_replay(heads, expected):
    """`--mamba-backend flashinfer` selects Mamba-2's FlashInfer ReplaySSM
    (its page alignment and autotune runs) only on models without GDP layers."""
    config = _vllm_config(heads=heads)
    config.mamba_config = SimpleNamespace(backend=MambaBackendEnum.FLASHINFER)
    config.use_gdp_replayssm = VllmConfig.use_gdp_replayssm.fget(config)
    assert VllmConfig.use_flashinfer_replayssm.fget(config) is expected
