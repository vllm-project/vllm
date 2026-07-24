# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import OPTConfig

from vllm.config.mamba import (
    MambaConfig,
    MambaDecodeBackendEnum,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    selective_state_update,
)
from vllm.model_executor.models.config import MambaModelConfig
from vllm.platforms.cpu import CpuPlatform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


def _kv_cache_config_with_ssu(
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2,
) -> KVCacheConfig:
    spec = MambaSpec(
        block_size=16,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_type=mamba_type,
    )
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
    )


def test_mamba_decode_and_prefill_cli(monkeypatch, caplog_vllm):
    monkeypatch.setattr("vllm.platforms.current_platform", CpuPlatform())
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.mamba_decode_backend is None
    assert defaults.mamba_prefill_backend is None

    canonical = parser.parse_args(["--mamba-decode-backend", "flashinfer"])
    assert canonical.mamba_decode_backend == "flashinfer"

    args = parser.parse_args(
        [
            "--mamba-backend",
            "flashinfer",
            "--mamba-prefill-backend",
            "flashinfer",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.mamba_decode_backend == "flashinfer"
    assert engine_args.mamba_prefill_backend == "flashinfer"
    assert "mamba_backend" in caplog_vllm.text
    assert "deprecated" in caplog_vllm.text


@pytest.mark.parametrize(
    ("flat_backend", "expected"), [(None, "flashinfer"), ("triton", "triton")]
)
def test_mamba_prefill_backend_override(monkeypatch, tmp_path, flat_backend, expected):
    platform = CpuPlatform()
    monkeypatch.setattr("vllm.platforms.current_platform", platform)
    monkeypatch.setattr("vllm.engine.arg_utils.current_platform", platform)
    OPTConfig().save_pretrained(tmp_path)
    config = EngineArgs(
        model=str(tmp_path),
        mamba_config=MambaConfig(prefill_backend="flashinfer"),
        mamba_prefill_backend=flat_backend,
    ).create_engine_config()
    assert config.mamba_config.prefill_backend.value == expected


def test_flashinfer_prefill_cache_mode():
    cache_config = SimpleNamespace(
        enable_prefix_caching=True,
        mamba_cache_mode="none",
        mamba_block_size=None,
        block_size=16,
    )
    config = SimpleNamespace(
        mamba_config=MambaConfig(prefill_backend="flashinfer"),
        model_config=SimpleNamespace(
            supports_mamba_prefix_caching=True,
            architecture="Mamba2ForCausalLM",
        ),
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    MambaModelConfig.verify_and_update_config(config)
    assert cache_config.mamba_cache_mode == "align"

    cache_config.mamba_cache_mode = "all"
    with pytest.raises(ValueError, match="does not support.*'all'"):
        MambaModelConfig.verify_and_update_config(config)

    config.model_config.supports_mamba_prefix_caching = False
    MambaModelConfig.verify_and_update_config(config)
    assert cache_config.mamba_cache_mode == "align"


def test_default_decode_backend_is_triton():
    initialize_mamba_ssu_backend(MambaConfig(), _kv_cache_config_with_ssu())
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)
    assert backend.name == "triton"


def test_explicit_triton_decode_backend():
    initialize_mamba_ssu_backend(
        MambaConfig(decode_backend=MambaDecodeBackendEnum.TRITON),
        _kv_cache_config_with_ssu(),
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_decode_backend_init():
    initialize_mamba_ssu_backend(
        MambaConfig(decode_backend=MambaDecodeBackendEnum.FLASHINFER),
        _kv_cache_config_with_ssu(),
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, FlashInferSSUBackend)
    assert backend.name == "flashinfer"


def test_uninitialized_backend_raises():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    with pytest.raises(RuntimeError, match="not been initialized"):
        get_mamba_ssu_backend()
    mod._mamba_ssu_backend = old


@pytest.mark.parametrize(
    "mamba_type",
    [
        MambaAttentionBackendEnum.LINEAR,
        MambaAttentionBackendEnum.GDN_ATTN,
        MambaAttentionBackendEnum.SHORT_CONV,
    ],
)
def test_init_is_noop_for_non_ssu_mamba_type(mamba_type):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    try:
        initialize_mamba_ssu_backend(
            MambaConfig(), _kv_cache_config_with_ssu(mamba_type)
        )
        assert mod._mamba_ssu_backend is None
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_mamba_ssu_backend()
    finally:
        mod._mamba_ssu_backend = old


@pytest.mark.skipif(HAS_FLASHINFER, reason="flashinfer is installed")
def test_flashinfer_import_error():
    with pytest.raises(ImportError, match="FlashInfer is required"):
        FlashInferSSUBackend(MambaConfig())


def test_triton_basic_call():
    set_random_seed(0)
    initialize_mamba_ssu_backend(
        MambaConfig(decode_backend=MambaDecodeBackendEnum.TRITON),
        _kv_cache_config_with_ssu(),
    )
    device = "cuda"
    batch_size = 2
    dim = 64
    dstate = 16

    state = torch.randn(batch_size, dim, dstate, device=device)
    x = torch.randn(batch_size, dim, device=device)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device)
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
    )
    assert not torch.isnan(out).any()
