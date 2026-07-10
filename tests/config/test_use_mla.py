# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `ModelConfig.use_mla` implementation-aware routing.

`use_mla` is not a pure config property: whether the MLA path is taken depends
on the selected implementation. Native code takes it only for the curated
`is_deepseek_mla` architectures (some MLA-shaped configs such as MiniCPM3 run as
standard attention natively), while the Transformers backend runs any
compressed-KV model through `MLAFuser`.
"""

from types import SimpleNamespace

import vllm.config.model as model_mod
from vllm.config.model import ModelConfig


def _use_mla(
    *,
    is_deepseek_mla: bool,
    transformers_backend: bool,
    kv_lora_rank: int | None,
    has_model_info: bool = True,
) -> bool:
    """Evaluate the `use_mla` property against a minimal stub."""
    stub = SimpleNamespace(
        is_deepseek_mla=is_deepseek_mla,
        _model_info=object() if has_model_info else None,
        using_transformers_backend=lambda: transformers_backend,
        hf_text_config=SimpleNamespace(kv_lora_rank=kv_lora_rank),
    )
    return ModelConfig.use_mla.fget(stub)


def test_native_mla_architecture_uses_mla():
    # DeepSeek native: allowlisted -> MLA, regardless of backend.
    assert _use_mla(is_deepseek_mla=True, transformers_backend=False, kv_lora_rank=512)


def test_native_mla_shaped_but_unlisted_stays_mha():
    # MiniCPM3 native: MLA-shaped config, but not allowlisted and not the
    # Transformers backend -> standard attention (no regression).
    assert not _use_mla(
        is_deepseek_mla=False, transformers_backend=False, kv_lora_rank=512
    )


def test_transformers_backend_mla_shaped_uses_mla():
    # Any compressed-KV model via the Transformers backend -> MLA (MLAFuser),
    # even when the architecture is not natively listed.
    assert _use_mla(is_deepseek_mla=False, transformers_backend=True, kv_lora_rank=512)


def test_transformers_backend_without_kv_lora_stays_mha():
    # No `kv_lora_rank` -> not MLA even on the Transformers backend.
    assert not _use_mla(
        is_deepseek_mla=False, transformers_backend=True, kv_lora_rank=None
    )


def test_model_info_not_yet_resolved_is_safe():
    # `using_transformers_backend` needs `_model_info`; guard against early reads.
    assert not _use_mla(
        is_deepseek_mla=False,
        transformers_backend=True,
        kv_lora_rank=512,
        has_model_info=False,
    )


def test_mla_disable_env_overrides_both_paths(monkeypatch):
    monkeypatch.setattr(model_mod.envs, "VLLM_MLA_DISABLE", True)
    assert not _use_mla(
        is_deepseek_mla=True, transformers_backend=False, kv_lora_rank=512
    )
    assert not _use_mla(
        is_deepseek_mla=False, transformers_backend=True, kv_lora_rank=512
    )
