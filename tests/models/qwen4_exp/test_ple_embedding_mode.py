# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Qwen4Exp `ple_embedding_mode` (ngram / per_token)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

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
from vllm.models.qwen4_exp.amd import ple_layer as amd_ple_layer
from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPerTokenEmbedding
from vllm.models.qwen4_exp.common.ple_mode import (
    validate_ple_embedding_config,
)

from .test_ple import _build_amd_ngram_embedding

VOCAB = 64
DIM = 6  # ngram_size=3, heads_per_ngram=1 -> 2 heads of dim 3


def _config(**kwargs) -> Qwen4ExpTextConfig:
    values = {
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "ngram_vocab_size_base": 5,
        "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 2,
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
    config = _config(ple_embedding_mode="per_token")
    return amd_ple_layer._build_ple_embedding(
        validate_ple_embedding_config(config),
        config,
        ple_dense_layer_id=1,
        max_num_batched_tokens=4,
        prefix="model.layers.1",
        use_sequence_parallel=use_sequence_parallel,
        quant_config=None,
        params_dtype=dtype,
        data_parallel_rank=0,
    )


# ---------------------------------------------------------------- config


def test_default_mode_is_ngram() -> None:
    assert validate_ple_embedding_config(_config()) == "ngram"


def test_per_token_validation_needs_no_ngram_fields() -> None:
    fields = {"ple_layer_ids": [1], "ple_embed_dim": 7, "vocab_size": VOCAB}
    assert (
        validate_ple_embedding_config(
            SimpleNamespace(ple_embedding_mode="per_token", **fields)
        )
        == "per_token"
    )
    with pytest.raises(ValueError, match="ple_embed_dim must be positive"):
        validate_ple_embedding_config(
            SimpleNamespace(
                ple_embedding_mode="per_token", **{**fields, "ple_embed_dim": 0}
            )
        )


def test_per_token_requires_single_ple_layer() -> None:
    with pytest.raises(ValueError, match="exactly one PLE layer"):
        validate_ple_embedding_config(
            _config(ple_embedding_mode="per_token", ple_layer_ids=[1, 3])
        )


# ---------------------------------------------------------------- forward


def test_per_token_forward_matches_manual_indexing() -> None:
    module = _per_token()
    table = _table()
    assert module.load_weights([("per_token_table.weight", table)]) == {
        "per_token_table.weight"
    }
    ids = torch.tensor([3, 0, 63, 3, 17], dtype=torch.int32)
    out = module(ids, None, None)
    assert out.shape == (5, DIM)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, table[ids.long()])
    # query_start_loc / ngram_context are ignored; 2D ids are flattened.
    out2 = module(
        ids.reshape(1, -1), torch.tensor([0, 5]), torch.zeros(1, 2, dtype=torch.int32)
    )
    assert torch.equal(out2, out)
    assert module.dequantize(out, torch.float16) is out


def test_per_token_out_of_range_ids_are_clamped() -> None:
    module = _per_token()
    table = _table()
    module.load_weights([("per_token_table.weight", table)])
    ids = torch.tensor([-5, -1, 0, VOCAB - 1, VOCAB, VOCAB + 1000], dtype=torch.int64)
    out = module(ids)
    expected = table[torch.tensor([0, 0, 0, VOCAB - 1, VOCAB - 1, VOCAB - 1])]
    assert torch.equal(out, expected)


def test_forward_is_fullgraph_compilable_with_dynamic_token_count() -> None:
    """No data-dependent host branches: fullgraph trace, then reuse for new sizes."""
    module = _per_token()
    table = _table()
    module.load_weights([("per_token_table.weight", table)])
    compiled = torch.compile(module, fullgraph=True, backend="eager", dynamic=True)
    for n in (7, 3):
        ids = torch.randint(0, VOCAB, (n,))
        assert torch.equal(compiled(ids), module(ids))


# ---------------------------------------------------------------- loader

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
    with pytest.raises(ValueError, match="dtype mismatch"):
        module.load_weights([("per_token_table.weight", _table().int())])
    # bf16 checkpoint into an fp16 model is the one allowed cast
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
    """Names as in the checkpoint, below the PLE layer (`ple_embedding.*`)."""

    class _PleOwner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ple_embedding = _per_token()
            self.other = nn.Linear(2, 2, bias=False)

    owner = _PleOwner()
    weights = [(f"ple_embedding.{n}", w) for n, w in _NGRAM_CHECKPOINT_TENSORS]
    weights.append(("ple_embedding.per_token_table.weight", _table()))
    weights.append(("other.weight", torch.eye(2)))
    loaded = AutoWeightsLoader(owner).load_weights(weights)
    assert torch.equal(owner.ple_embedding.per_token_table.weight, _table())
    assert loaded == {"other.weight", "ple_embedding.per_token_table.weight"}


# ---------------------------------------------------------------- ngram mode unchanged


def test_ngram_mode_skips_per_token_checkpoint_tensor(monkeypatch) -> None:
    """A checkpoint carrying both tables can be served in either mode."""
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
    from vllm.platforms import current_platform

    config = SimpleNamespace(
        cache_config=CacheConfig(),
        model_config=SimpleNamespace(
            hf_text_config=_config(ple_embedding_mode=mode, hc_count=2)
        ),
    )
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    with pytest.raises(error, match=message):
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(config)


def test_hf_overrides_text_config_dict_sets_mode() -> None:
    """Same call as ModelConfig: hf_overrides={"text_config": {...}}."""
    from transformers import Qwen4ExpConfig

    from vllm.config.model import ModelConfig

    config = Qwen4ExpConfig(text_config=_config().to_dict())
    model_config = object.__new__(ModelConfig)
    model_config._apply_dict_overrides(
        config, {"text_config": {"ple_embedding_mode": "per_token"}}
    )
    assert validate_ple_embedding_config(config.get_text_config()) == "per_token"
    assert config.get_text_config().vocab_size == VOCAB  # other fields kept
    # A top-level override would be a silent no-op for the layer.
    config = Qwen4ExpConfig(text_config=_config().to_dict())
    config.update({"ple_embedding_mode": "per_token"})
    assert validate_ple_embedding_config(config.get_text_config()) == "ngram"


def _fill_load_weights(module: Qwen4ExpPerTokenEmbedding) -> None:
    module.load_weights([("per_token_table.weight", _table())])


def _fill_direct_copy(module: Qwen4ExpPerTokenEmbedding) -> None:
    """What sharded_state does: copy into state_dict() without load_weights."""
    module.state_dict()["per_token_table.weight"].copy_(_table())


def _fill_dummy(module: Qwen4ExpPerTokenEmbedding) -> None:
    for name, param in module.named_parameters():
        initialize_single_dummy_weight(param.data, name=name)


@pytest.mark.parametrize(
    "fill",
    [_fill_load_weights, _fill_direct_copy, _fill_dummy],
    ids=["load_weights", "direct_copy", "dummy"],
)
def test_per_token_post_load_check_follows_the_table_contents(fill) -> None:
    module = _per_token()
    with pytest.raises(ValueError, match="never loaded"):
        finalize_model_weights(module)
    fill(module)
    finalize_model_weights(module)
    weight = module.per_token_table.weight
    if fill is _fill_dummy:
        assert torch.isfinite(weight).all()
    else:
        assert torch.equal(weight, _table())


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
    """Weight updates reload layer by layer (the default `VLLM_RELOAD_MODE`)."""
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


# ---------------------------------------------------------------- factory


def test_factory_per_token_rejects_sequence_parallel() -> None:
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        _per_token(use_sequence_parallel=True)
