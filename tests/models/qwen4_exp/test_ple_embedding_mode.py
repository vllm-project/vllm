# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Qwen4Exp `ple_embedding_mode` (ngram / per_token)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config import CacheConfig
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.weight_utils import (
    finalize_model_weights,
    initialize_single_dummy_weight,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPerTokenEmbedding
from vllm.models.qwen4_exp.common.ple_mode import validate_ple_embedding_config
from vllm.platforms import current_platform

from .test_ple import _build_amd_ngram_embedding

VOCAB = 64
DIM = 6


def _config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "ple_embed_dim": DIM,
        "ple_layer_ids": [1],
        "eos_token_id": 0,
        "vocab_size": VOCAB,
    }
    values.update(kwargs)
    return Qwen4ExpTextConfig(**values)


def _table() -> torch.Tensor:
    return torch.arange(VOCAB * DIM, dtype=torch.float32).reshape(VOCAB, DIM).bfloat16()


def _per_token(dtype=torch.bfloat16, *, use_sequence_parallel=False):
    return Qwen4ExpPerTokenEmbedding(
        _config(ple_embedding_mode="per_token"), dtype, use_sequence_parallel
    )


def test_default_mode_is_ngram() -> None:
    assert validate_ple_embedding_config(_config()) == "ngram"


@pytest.mark.parametrize("field", ["ple_embed_dim", "vocab_size"])
def test_per_token_rejects_invalid_table_dimensions(field):
    config = _config(ple_embedding_mode="per_token")
    setattr(config, field, 0)
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        validate_ple_embedding_config(config)


def test_per_token_requires_single_ple_layer() -> None:
    with pytest.raises(ValueError, match="exactly one PLE layer"):
        validate_ple_embedding_config(
            _config(ple_embedding_mode="per_token", ple_layer_ids=[1, 3])
        )


def test_per_token_forward_matches_manual_indexing() -> None:
    module = _per_token()
    table = _table()
    assert module.load_weights([("per_token_table.weight", table)]) == {
        "per_token_table.weight"
    }
    ids = torch.tensor([[-1, 3, VOCAB, 3]], dtype=torch.int32)
    out = module(ids)
    assert out.dtype == table.dtype
    assert torch.equal(out, table[[0, 3, VOCAB - 1, 3]])


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="per-token PLE requires ROCm"
)
def test_per_token_runner_omits_ngram_capture_inputs(tmp_path) -> None:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

    Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=_config().to_dict(),
    ).save_pretrained(tmp_path)
    config = EngineArgs(
        model=str(tmp_path),
        language_model_only=True,
        hf_overrides={"text_config": {"ple_embedding_mode": "per_token"}},
        skip_tokenizer_init=True,
        max_model_len=128,
    ).create_engine_config()
    state = Qwen4ExpModelState(config, nn.Module(), None, torch.device("cpu"))
    inputs = state.prepare_dummy_inputs(num_reqs=2, num_tokens=4)
    assert {"query_start_loc", "ngram_context"}.isdisjoint(inputs)


def test_forward_is_fullgraph_compilable_with_dynamic_token_count() -> None:
    module = _per_token()
    table = _table()
    module.load_weights([("per_token_table.weight", table)])
    compiled = torch.compile(module, fullgraph=True, backend="eager", dynamic=True)
    for n in (7, 3):
        ids = torch.randint(0, VOCAB, (n,))
        assert torch.equal(compiled(ids), module(ids))


_NGRAM_CHECKPOINT_TENSORS = [
    ("layer_multipliers", torch.ones(3, dtype=torch.long)),
    ("ngram_heads_offsets", torch.zeros(2, dtype=torch.long)),
    ("ngram_heads_vocab_sizes", torch.ones(2, dtype=torch.long)),
    ("ngram_embedding.shard_0.weight", torch.zeros(4, 3)),
    ("ngram_embedding.shard_1.weight", torch.zeros(4, 3)),
    ("hashstats_x", torch.zeros(1)),
]


def test_per_token_loader_dtype_policy() -> None:
    module = _per_token()
    with pytest.raises(ValueError, match="dtype mismatch"):
        module.load_weights([("per_token_table.weight", _table().float())])
    half = _per_token(dtype=torch.float16)
    half.load_weights([("per_token_table.weight", _table())])
    assert half.per_token_table.weight.dtype == torch.float16
    assert torch.equal(half.per_token_table.weight, _table().half())


def test_per_token_loader_rejects_wrong_shape_and_unknown_names() -> None:
    module = _per_token()
    with pytest.raises(ValueError, match="Shape mismatch"):
        module.load_weights([("per_token_table.weight", torch.zeros(VOCAB - 1, DIM))])
    with pytest.raises(ValueError, match="Unexpected PLE embedding tensor"):
        module.load_weights([("bogus.weight", torch.zeros(1))])


def test_loader_through_auto_weights_loader_with_checkpoint_names() -> None:
    owner = nn.Module()
    owner.ple_embedding = _per_token()
    weights = [(f"ple_embedding.{n}", w) for n, w in _NGRAM_CHECKPOINT_TENSORS]
    weights.append(("ple_embedding.per_token_table.weight", _table()))
    loaded = AutoWeightsLoader(owner).load_weights(weights)
    assert torch.equal(owner.ple_embedding.per_token_table.weight, _table())
    assert loaded == {"ple_embedding.per_token_table.weight"}


def test_ngram_mode_skips_per_token_checkpoint_tensor(monkeypatch) -> None:
    module, loaded = _build_amd_ngram_embedding(
        monkeypatch, device="cpu", cpu_offload=False, fp8_checkpoint=False
    )
    assert module.load_weights([("per_token_table.weight", _table())]) == set()
    embedding = module.ngram_embedding
    assert torch.equal(embedding.weight[: embedding.org_vocab_size], loaded)
    with pytest.raises(ValueError):
        module.load_weights([("unknown_table.weight", _table())])


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ("bogus", ValueError, "Invalid ple_embedding_mode"),
        ("per_token", NotImplementedError, "only implemented for ROCm"),
    ],
)
def test_model_config_rejects_unsupported_ple_mode(monkeypatch, mode, error, message):
    from vllm.model_executor.models.config import Qwen4ExpForConditionalGenerationConfig

    config = SimpleNamespace(
        cache_config=CacheConfig(),
        model_config=SimpleNamespace(
            hf_text_config=_config(ple_embedding_mode=mode, hc_count=2)
        ),
    )
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    with pytest.raises(error, match=message):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)


@pytest.mark.parametrize("dummy", [False, True], ids=["sharded_state", "dummy"])
def test_post_load_check_accepts_initialization_without_weight_loader(dummy) -> None:
    module = _per_token()
    with pytest.raises(ValueError, match="never loaded"):
        finalize_model_weights(module)
    weight = module.state_dict()["per_token_table.weight"]
    if dummy:
        initialize_single_dummy_weight(weight)
    else:
        weight.copy_(_table())
    finalize_model_weights(module)


def test_per_token_nan_in_loaded_table_fails() -> None:
    module = _per_token()
    table = _table()
    table[3, 2] = float("nan")
    module.load_weights([("per_token_table.weight", table)])
    with pytest.raises(ValueError, match="NaN"):
        finalize_model_weights(module)


@pytest.mark.parametrize("update_has_table", [True, False])
def test_per_token_table_through_a_layerwise_weight_update(
    monkeypatch: pytest.MonkeyPatch, update_has_table: bool
) -> None:
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    module = _per_token()
    record_metadata_for_reloading(module)
    module.load_weights([("per_token_table.weight", _table())])
    finalize_model_weights(module)

    start_reload(module)
    update = [("per_token_table.weight", 2 * _table())] if update_has_table else []
    module.load_weights(update + _NGRAM_CHECKPOINT_TENSORS)
    finish_reload(module, model_config=None, device=torch.device("cpu"))

    expected = 2 * _table() if update_has_table else _table()
    assert torch.equal(module.per_token_table.weight, expected)


def test_per_token_rejects_sequence_parallel() -> None:
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        _per_token(use_sequence_parallel=True)
