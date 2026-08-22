# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 checkpoints that ship a DSpark drafter must not be routed to MTP.

``deepseek-ai/DeepSeek-V4-Flash-0731`` advertises ``num_nextn_predict_layers: 1``
like every other DeepSeek-V4 config, but the ``mtp.*`` tensors behind it are a
three-stage DSpark drafter with no ``enorm``/``hnorm``/``e_proj``/``h_proj``.
Routing it to ``DeepSeekV4MTPModel`` fails deep inside the weight loader with
``KeyError: model.layers.43.mtp_block.main_norm.weight`` (vllm-project/vllm#52111).

Which DSpark drafter a config describes was previously re-derived from
architecture strings at four sites; ``DSparkVariant`` resolves it once, so these
tests also pin the variant mapping those sites now share.
"""

import json

import pytest
from transformers import PretrainedConfig

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import (
    DSparkVariant,
    SpeculativeConfig,
    _is_deepseek_v4_dspark,
    _is_dspark_draft,
)

# Trimmed from the published config.json. Both DeepSeek-V4-Flash variants
# declare num_hidden_layers=43 and num_nextn_predict_layers=1, so only the
# dspark_* keys below tell the DSpark drafter apart from a real MTP head.
_DEEPSEEK_V4 = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "vocab_size": 129280,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
}

_DSPARK_KEYS = {
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
    "dspark_noise_token_id": 128799,
}

# DeepSeek-V4-Pro-0813, the other shipped DSpark checkpoint: deeper, with
# different target layers and markov rank, and the same
# num_nextn_predict_layers=1. Detection must depend on neither the layer count
# nor the particular ids.
_PRO_DSPARK = {
    "num_hidden_layers": 61,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [58, 59, 60],
    "dspark_markov_rank": 512,
    "dspark_noise_token_id": 128799,
}


def _hf_config(**kwargs) -> PretrainedConfig:
    config = PretrainedConfig(**{**_DEEPSEEK_V4, **kwargs})
    config.model_type = kwargs.get("model_type", "deepseek_v4")
    return config


def _checkpoint(tmp_path, name: str, **extra) -> ModelConfig:
    """A real ``ModelConfig`` over a synthetic checkpoint directory.

    Only ``config.json`` is written: routing is decided from the config alone,
    so no weights and no network access are needed.
    """
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps({**_DEEPSEEK_V4, **extra}))
    return ModelConfig(
        model=str(path),
        tokenizer_mode="skip",
        skip_tokenizer_init=True,
        max_model_len=4096,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "name,keys",
    [("DeepSeek-V4-Flash-0731", _DSPARK_KEYS), ("DeepSeek-V4-Pro-0813", _PRO_DSPARK)],
)
def test_explicit_mtp_on_dspark_checkpoint_is_rejected(tmp_path, name, keys):
    """The reported invocation must fail at config time, not in the workers.

    Both checkpoints crash in the weight loader without this guard, at
    ``model.layers.{43,61}.mtp_block.main_norm.weight`` respectively.
    """
    target = _checkpoint(tmp_path, name, **keys)

    with pytest.raises(ValueError, match="ships a DSpark drafter"):
        SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=1,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
def test_explicit_mtp_is_rejected_for_a_named_draft_model(tmp_path):
    """``hf_config_override`` hides the model_type, so the draft path needs its
    own check; without it the explicit request is silently rewritten to DSpark."""
    target = _checkpoint(tmp_path, "DeepSeek-V4-Flash-0731", **_DSPARK_KEYS)

    with pytest.raises(ValueError, match="ships a DSpark drafter"):
        SpeculativeConfig(
            method="mtp",
            model=target.model,
            num_speculative_tokens=1,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
def test_rejection_names_the_token_count_dspark_needs(tmp_path):
    """The suggested method has its own minimum; say so in one error, not two."""
    target = _checkpoint(tmp_path, "DeepSeek-V4-Flash-0731", **_DSPARK_KEYS)

    with pytest.raises(ValueError, match=r"dspark_block_size \(5\)"):
        SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=1,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
def test_plain_mtp_checkpoint_still_routes_to_mtp(tmp_path):
    """DeepSeek-V4-Flash has a real MTP head and must be left alone."""
    target = _checkpoint(tmp_path, "DeepSeek-V4-Flash")

    spec = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=1,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )

    assert spec.method == "mtp"
    assert spec.draft_model_config.hf_config.architectures == ["DeepSeekV4MTPModel"]


@pytest.mark.cpu_test
def test_omitted_method_auto_detects_dspark(tmp_path):
    """Detection must reach DSpark from the config, not from the repo name."""
    target = _checkpoint(tmp_path, "DeepSeek-V4-Flash-0731", **_DSPARK_KEYS)

    spec = SpeculativeConfig(
        model=target.model,
        num_speculative_tokens=5,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )

    assert spec.method == "dspark"
    assert spec.draft_model_config.hf_config.architectures == ["DSparkDraftModel"]


@pytest.mark.cpu_test
def test_explicit_dspark_is_accepted(tmp_path):
    target = _checkpoint(tmp_path, "DeepSeek-V4-Flash-0731", **_DSPARK_KEYS)

    spec = SpeculativeConfig(
        method="dspark",
        num_speculative_tokens=5,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )

    assert spec.method == "dspark"
    assert spec.draft_model_config.hf_config.architectures == ["DSparkDraftModel"]


@pytest.mark.cpu_test
@pytest.mark.parametrize("keys", [_DSPARK_KEYS, _PRO_DSPARK])
def test_dspark_drafter_detected_from_config_keys(keys):
    """Both shipped DSpark checkpoints, whose layer counts and target layer ids
    differ while num_nextn_predict_layers does not."""
    assert _is_deepseek_v4_dspark(_hf_config(**keys))
    assert not _is_deepseek_v4_dspark(_hf_config())


@pytest.mark.cpu_test
def test_empty_target_layer_ids_is_not_a_drafter():
    """A drafter with no target layers cannot be loaded; do not claim it."""
    assert not _is_deepseek_v4_dspark(_hf_config(dspark_target_layer_ids=[]))


@pytest.mark.cpu_test
@pytest.mark.parametrize("model_type", ["deepseek_v3", "deepseek_v32", "qwen3_next"])
def test_other_model_types_are_untouched(model_type):
    config = _hf_config(**_DSPARK_KEYS)
    config.model_type = model_type

    assert not _is_deepseek_v4_dspark(config)


@pytest.mark.cpu_test
def test_detected_after_hf_config_override():
    """The draft path sees the config after ``hf_config_override`` has rewritten
    model_type to ``deepseek_mtp``; the dspark_* keys survive, so detection must
    survive with them."""
    overridden = SpeculativeConfig.hf_config_override(_hf_config(**_DSPARK_KEYS))

    assert overridden.model_type == "deepseek_mtp"
    assert overridden.architectures == ["DeepSeekV4MTPModel"]
    assert _is_deepseek_v4_dspark(overridden)

    plain = SpeculativeConfig.hf_config_override(_hf_config())
    assert plain.architectures == ["DeepSeekV4MTPModel"]
    assert not _is_deepseek_v4_dspark(plain)


@pytest.mark.cpu_test
def test_dspark_draft_detected_without_dspark_in_the_name():
    """Auto-detection must not depend on the repo being named ``*dspark*``."""
    assert _is_dspark_draft(
        "deepseek-ai/DeepSeek-V4-Flash-0731", _hf_config(**_DSPARK_KEYS)
    )
    assert not _is_dspark_draft("deepseek-ai/DeepSeek-V4-Flash", _hf_config())


@pytest.mark.cpu_test
def test_dspark_draft_still_detected_by_name():
    """The name remains a fallback for checkpoints that declare nothing."""
    assert _is_dspark_draft("deepseek-ai/dspark_qwen3_8b_block7", PretrainedConfig())


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "architecture,expected",
    [
        ("Qwen3DSparkModel", DSparkVariant.QWEN3),
        ("Gemma4DSparkModel", DSparkVariant.GEMMA4),
        ("K3DSparkModel", DSparkVariant.K3),
    ],
)
def test_variant_resolved_from_declared_architecture(architecture, expected):
    config = PretrainedConfig(architectures=[architecture])

    assert DSparkVariant.from_config(config) is expected


@pytest.mark.cpu_test
def test_synthesised_architecture_with_qwen3_resolves_to_qwen3():
    """A Qwen3 DSpark draft may declare the synthesised `DSparkDraftModel`
    name (#52197). Without the `model_type` pairing it would fall through to
    DEEPSEEK_V4 and have its `model_type` rewritten to `deepseek_v4`."""
    config = PretrainedConfig(architectures=["DSparkDraftModel"], model_type="qwen3")

    assert DSparkVariant.from_config(config) is DSparkVariant.QWEN3
    assert _is_dspark_draft("some/qwen3-draft", config)


@pytest.mark.cpu_test
def test_k3_draft_is_not_auto_routed_to_dspark():
    """K3 declares a DSpark architecture but upstream leaves it a plain draft
    model unless the method is explicit; detection must not widen that."""
    config = PretrainedConfig(architectures=["K3DSparkModel"], model_type="k3_dspark")

    assert not _is_dspark_draft("Inferact/Kimi-K3", config)
    assert DSparkVariant.from_config(config) is DSparkVariant.K3


@pytest.mark.cpu_test
def test_deepseek_v4_is_the_variant_without_its_own_architecture():
    """DeepSeek-V4 DSpark reuses the target's config, so it declares no draft
    architecture of its own and is what remains once the others are excluded."""
    assert DSparkVariant.from_config(_hf_config(**_DSPARK_KEYS)) is (
        DSparkVariant.DEEPSEEK_V4
    )
    assert DSparkVariant.DEEPSEEK_V4.value == "DSparkDraftModel"
