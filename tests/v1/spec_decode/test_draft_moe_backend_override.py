# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft model must honour ``moe_backend`` from --speculative-config.

The draft is loaded with the target's VllmConfig, so without an explicit
override it inherits the target's --moe-backend. An MTP head on a quantized
target is typically unquantized, and quantized-only backends reject it, so the
server fails to start rather than falling back.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers import Qwen2Config

from vllm import SamplingParams
from vllm.config import (
    KernelConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _KernelConfig:
    moe_backend: str | None = None


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _SpeculativeConfig:
    attention_backend: str | None = None
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    draft_model_config: object = None
    draft_load_config: object = None

    apply_draft_overrides = SpeculativeConfig.apply_draft_overrides


@dataclass
class _VllmConfig:
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)


def _config(target_moe: str, draft_moe: str | None) -> _VllmConfig:
    return _VllmConfig(
        kernel_config=_KernelConfig(moe_backend=target_moe),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(moe_backend=draft_moe),
    )


class _Captured(Exception):
    """Stops load_eagle_model once we hold the config it would have used."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def _capture_draft_config(cfg):
    def _fake_get_model(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    with (
        patch("vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model", _fake_get_model),
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), cfg)
    return exc.value.vllm_config


def test_draft_moe_backend_overrides_the_target():
    """A draft moe_backend must reach the draft's kernel config."""
    used = _capture_draft_config(_config("flashinfer_b12x", "flashinfer_cutlass"))
    assert used.kernel_config.moe_backend == "flashinfer_cutlass"


def test_draft_inherits_target_when_no_override():
    """No override means the draft still inherits the target, as before."""
    used = _capture_draft_config(_config("flashinfer_b12x", None))
    assert used.kernel_config.moe_backend == "flashinfer_b12x"


def test_override_does_not_mutate_the_target_config():
    """The target must keep its own backend after the draft is built."""
    cfg = _config("flashinfer_b12x", "flashinfer_cutlass")
    _capture_draft_config(cfg)
    assert cfg.kernel_config.moe_backend == "flashinfer_b12x"


@pytest.mark.parametrize(
    "target_backend,draft_backend,all2all,expected",
    [
        (
            "flashinfer_moe_ep_cutedsl",
            "flashinfer_trtllm",
            "allgather_reducescatter",
            "allgather_reducescatter",
        ),
        ("flashinfer_moe_ep_cutedsl", None, "allgather_reducescatter", "passthrough"),
        (
            "flashinfer_moe_ep_cutedsl",
            "flashinfer_moe_ep_cutedsl",
            "allgather_reducescatter",
            "passthrough",
        ),
        (
            "flashinfer_trtllm",
            "flashinfer_cutlass",
            "deepep_low_latency",
            "deepep_low_latency",
        ),
    ],
)
def test_draft_override_resolves_its_own_communication(
    tmp_path, target_backend, draft_backend, all2all, expected
):
    """A regular draft must not inherit its target's inferred passthrough."""
    Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    ).save_pretrained(tmp_path)
    target = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), skip_tokenizer_init=True),
        kernel_config=KernelConfig(moe_backend=target_backend),
        parallel_config=ParallelConfig(all2all_backend=all2all),
    )
    original_all2all = target.parallel_config.all2all_backend
    spec = SpeculativeConfig(
        method="ngram", num_speculative_tokens=2, moe_backend=draft_backend
    )
    draft = spec.apply_draft_overrides(target)
    assert draft.kernel_config.moe_backend == (draft_backend or target_backend)
    assert draft.parallel_config.all2all_backend == expected
    assert target.kernel_config.moe_backend == target_backend
    assert target.parallel_config.all2all_backend == original_all2all


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA backend override")
def test_mtp_with_self_dispatching_target_backend(vllm_runner, tmp_path):
    """An explicit regular draft backend can load and generate with MTP."""
    Qwen2Config(
        architectures=["MiMoForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_nextn_predict_layers=1,
        vocab_size=1024,
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        kernel_config={
            "moe_backend": "flashinfer_moe_ep_cutedsl",
            "enable_jit_warmup": False,
        },
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 2,
            "moe_backend": "flashinfer_trtllm",
        },
    ) as runner:
        (output,) = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3]}],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
        )
        tokens = output.outputs[0].token_ids
        assert len(tokens) == 4
        assert all(0 <= token < 1024 for token in tokens)
