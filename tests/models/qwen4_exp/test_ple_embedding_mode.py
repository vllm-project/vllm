# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Qwen4Exp `ple_embedding_mode` (ngram / per_token)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
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
from vllm.model_executor.utils import weights_already_processed
from vllm.models.qwen4_exp.amd import ple_layer as amd_ple_layer
from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPerTokenEmbedding
from vllm.models.qwen4_exp.common.ple_mode import (
    PLE_EMBEDDING_MODES,
    get_ple_embedding_mode,
    validate_ple_embedding_config,
)

from .test_ple import _build_amd_ngram_embedding, _mock_etp_group

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


def _per_token(dtype=torch.bfloat16):
    return Qwen4ExpPerTokenEmbedding(
        _config(ple_embedding_mode="per_token"), DIM, params_dtype=dtype
    )


# ---------------------------------------------------------------- config


def test_default_mode_is_ngram() -> None:
    assert get_ple_embedding_mode(_config()) == "ngram"
    assert validate_ple_embedding_config(_config()) == "ngram"
    assert PLE_EMBEDDING_MODES == ("ngram", "per_token")


@pytest.mark.parametrize("mode", PLE_EMBEDDING_MODES)
def test_mode_is_carried_through_config_dict(mode: str) -> None:
    config = _config(ple_embedding_mode=mode)
    assert get_ple_embedding_mode(config) == mode
    restored = Qwen4ExpTextConfig.from_dict(config.to_dict())
    assert get_ple_embedding_mode(restored) == mode
    assert validate_ple_embedding_config(restored) == mode


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid ple_embedding_mode"):
        validate_ple_embedding_config(_config(ple_embedding_mode="unigram"))


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


def test_unchanged_original_checkpoint_config_values() -> None:
    config = _config(
        ple_embed_dim=2560,
        heads_per_ngram=8,
        vocab_size=248320,
        ple_embedding_mode="per_token",
    )
    assert validate_ple_embedding_config(config) == "per_token"


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


def test_per_token_module_has_only_the_table() -> None:
    module = _per_token()
    assert list(module.state_dict()) == ["per_token_table.weight"]
    assert module.per_token_table.weight.shape == (VOCAB, DIM)
    assert module.per_token_table.weight.dtype == torch.bfloat16


def test_per_token_loader_skips_ngram_tensors() -> None:
    module = _per_token()
    table = _table()
    loaded = module.load_weights(
        _NGRAM_CHECKPOINT_TENSORS + [("per_token_table.weight", table)]
    )
    assert loaded == {"per_token_table.weight"}
    assert torch.equal(module.per_token_table.weight, table)


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


def test_per_token_loader_reports_missing_table() -> None:
    module = _per_token()
    assert module.load_weights(_NGRAM_CHECKPOINT_TENSORS) == set()


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


def test_ngram_embedding_still_loads_original_checkpoint_layout(monkeypatch) -> None:
    module, loaded_weight = _build_amd_ngram_embedding(
        monkeypatch, device="cpu", cpu_offload=False, fp8_checkpoint=False
    )
    embedding = module.ngram_embedding
    assert torch.equal(embedding.weight, loaded_weight)
    # The hash buffers stay persistent checkpoint tensors in ngram mode.
    assert {"layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"} <= (
        set(module.state_dict())
    )
    assert not hasattr(module, "per_token_table")
    # dequantize on the owner delegates to the embedding (identity for bf16).
    rows = embedding.weight[:2]
    assert module.dequantize(rows, torch.bfloat16) is rows


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


def test_verify_and_update_config_rejects_invalid_mode() -> None:
    from unittest.mock import patch

    from vllm.model_executor.models.config import (
        Qwen3_5ForConditionalGenerationConfig,
        Qwen4ExpForConditionalGenerationConfig,
    )

    def vllm_config(mode: str):
        text_config = _config(
            ple_embedding_mode=mode,
            hc_count=2,
            hc_lowrank=4,
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["linear_attention", "linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=text_config, multimodal_config=None
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1,
                enable_dbo=False,
                ubatch_size=1,
                use_sequence_parallel_moe=False,
            ),
            speculative_config=None,
        )

    rocm = SimpleNamespace(is_rocm=lambda: True, is_cuda_alike=lambda: True)
    other = SimpleNamespace(is_rocm=lambda: False, is_cuda_alike=lambda: False)
    with patch.object(
        Qwen3_5ForConditionalGenerationConfig, "verify_and_update_config"
    ):
        with patch("vllm.platforms.current_platform", rocm):
            for mode in PLE_EMBEDDING_MODES:
                Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                    vllm_config(mode)
                )
            with pytest.raises(ValueError, match="Invalid ple_embedding_mode"):
                Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                    vllm_config("bogus")
                )
        with patch("vllm.platforms.current_platform", other):
            Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                vllm_config("ngram")
            )
            with pytest.raises(NotImplementedError, match="only implemented"):
                Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(
                    vllm_config("per_token")
                )


@pytest.mark.parametrize("mode", PLE_EMBEDDING_MODES)
def test_mode_is_read_from_text_config_of_composite_config(mode: str) -> None:
    """hf_overrides text_config.ple_embedding_mode must reach the layer."""
    from transformers import Qwen4ExpConfig

    text = _config().to_dict()
    text["ple_embedding_mode"] = mode
    composite = Qwen4ExpConfig(text_config=text)
    assert get_ple_embedding_mode(composite.get_text_config()) == mode
    # A top-level field is NOT seen by the layer (it reads the text config).
    top = Qwen4ExpConfig(text_config=_config().to_dict(), ple_embedding_mode=mode)
    assert get_ple_embedding_mode(top.get_text_config()) == "ngram"


def test_hf_overrides_text_config_dict_sets_mode() -> None:
    """Same call as ModelConfig: hf_overrides={"text_config": {...}}."""
    from transformers import Qwen4ExpConfig

    from vllm.config.model import ModelConfig

    config = Qwen4ExpConfig(text_config=_config().to_dict())
    model_config = object.__new__(ModelConfig)
    model_config._apply_dict_overrides(
        config, {"text_config": {"ple_embedding_mode": "per_token"}}
    )
    assert get_ple_embedding_mode(config.get_text_config()) == "per_token"
    assert config.get_text_config().vocab_size == VOCAB  # other fields kept
    # A top-level override would be a silent no-op for the layer.
    config = Qwen4ExpConfig(text_config=_config().to_dict())
    config.update({"ple_embedding_mode": "per_token"})
    assert get_ple_embedding_mode(config.get_text_config()) == "ngram"


def test_flat_text_config_mode_and_overrides() -> None:
    """Flat text config (model_type qwen4_exp_text), field at top level."""
    config = _config(ple_embedding_mode="per_token")
    flat = Qwen4ExpTextConfig.from_dict(config.to_dict())
    assert flat.get_text_config() is flat
    assert get_ple_embedding_mode(flat) == "per_token"
    # hf_overrides={"ple_embedding_mode": ...} on a flat config (config.update)
    flat = _config()
    flat.update({"ple_embedding_mode": "per_token"})
    assert get_ple_embedding_mode(flat.get_text_config()) == "per_token"


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


def test_per_token_never_loaded_fails_loudly() -> None:
    module = _per_token()
    module.load_weights(_NGRAM_CHECKPOINT_TENSORS)  # table absent
    with pytest.raises(ValueError, match="per_token_table.weight"):
        finalize_model_weights(module)


def test_per_token_nan_in_loaded_table_fails() -> None:
    module = _per_token()
    table = _table()
    table[3, 2] = float("nan")
    module.load_weights([("per_token_table.weight", table)])
    with pytest.raises(ValueError, match="NaN"):
        finalize_model_weights(module)


def test_per_token_pre_processed_weights_skip_the_check() -> None:
    """IPC weight-cache loader runs the post-load pass under this context."""
    module = _per_token()
    with pytest.raises(ValueError, match="never loaded"):
        finalize_model_weights(module)
    with weights_already_processed():
        finalize_model_weights(module)


def test_per_token_meta_device_skips_the_check() -> None:
    with pytest.raises(ValueError, match="never loaded"):
        finalize_model_weights(_per_token())
    with torch.device("meta"):
        module = _per_token()
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


def _build(mode: str, *, use_sequence_parallel: bool = False) -> nn.Module:
    return amd_ple_layer._build_ple_embedding(
        mode,
        _config(ple_embedding_mode=mode),
        1,
        4,
        "model.layers.1",
        use_sequence_parallel,
        quant_config=None,
        params_dtype=torch.bfloat16,
        data_parallel_rank=0,
    )


@pytest.mark.parametrize(
    ("mode", "cls"),
    [
        ("per_token", Qwen4ExpPerTokenEmbedding),
        ("ngram", amd_ple_layer.Qwen4ExpNGramEmbedding),
    ],
)
def test_factory_builds_embedding_for_mode(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    mode: str,
    cls: type[nn.Module],
) -> None:
    # Same single-rank setup as _build_amd_ngram_embedding (needed by ngram only).
    _mock_etp_group(monkeypatch)
    for module in (embedding_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    module = _build(mode)
    assert type(module) is cls
    if mode == "per_token":
        assert module.per_token_table.weight.dtype == torch.bfloat16
        assert module.per_token_table.weight.shape == (VOCAB, DIM)


def test_factory_per_token_rejects_sequence_parallel() -> None:
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        _build("per_token", use_sequence_parallel=True)
