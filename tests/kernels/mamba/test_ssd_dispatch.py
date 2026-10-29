# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config.mamba import MambaConfig
from vllm.model_executor.layers.mamba.ops import ssd_dispatch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec


@pytest.fixture
def kv_cache_config(monkeypatch):
    monkeypatch.setattr(ssd_dispatch, "_mamba_ssd_backend", None)
    spec = MambaSpec(block_size=16, shapes=((8, 64, 128),), dtypes=(torch.float16,))
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["mixer"], kv_cache_spec=spec)],
    )


@pytest.mark.parametrize("explicit", [True, False])
def test_flashinfer_requirement_errors_but_inherited_preference_falls_back(
    monkeypatch, caplog_vllm, kv_cache_config, explicit
):
    monkeypatch.setitem(sys.modules, "flashinfer.mamba", SimpleNamespace())
    config = MambaConfig(
        backend="flashinfer",
        decode_backend="flashinfer",
        prefill_backend="flashinfer" if explicit else None,
    )
    original = asdict(config)
    if explicit:
        with pytest.raises(ImportError, match="ssd_combined_fwd"):
            ssd_dispatch.initialize_mamba_ssd_backend(config, kv_cache_config)
    else:
        with caplog_vllm.at_level(logging.WARNING):
            for _ in range(2):
                ssd_dispatch.initialize_mamba_ssd_backend(config, kv_cache_config)
                backend = ssd_dispatch.get_mamba_ssd_backend()
                assert isinstance(backend, ssd_dispatch.TritonSSDBackend)
        warnings = [
            record
            for record in caplog_vllm.records
            if record.name == ssd_dispatch.__name__
            and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
    assert asdict(config) == original


@pytest.mark.parametrize(
    "mamba_type",
    [None, MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.GDN_ATTN],
)
def test_initialize_ssd_ignores_cache_groups_without_mamba2(
    kv_cache_config, mamba_type
):
    """Other layers must not import an explicitly selected FlashInfer SSD."""
    if mamba_type is None:
        kv_cache_config.kv_cache_groups.clear()
    else:
        group = kv_cache_config.kv_cache_groups[0]
        group.kv_cache_spec = replace(group.kv_cache_spec, mamba_type=mamba_type)
    ssd_dispatch.initialize_mamba_ssd_backend(
        MambaConfig(prefill_backend="flashinfer"), kv_cache_config
    )
    assert ssd_dispatch._mamba_ssd_backend is None
