# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A checkpoint that declares bidirectional attention spans is served with them
or not at all."""

from unittest import mock

import pytest

from vllm.config import CacheConfig, KVTransferConfig, ModelConfig, VllmConfig

pytestmark = pytest.mark.cpu_test

MODEL = "google/gemma-4-E2B-it"
SPANS = {
    "bidirectional_span_layer_types": ["full_attention"],
    "bidirectional_span_max_length": 1024,
}


def span_model_config(**text_config) -> ModelConfig:
    return ModelConfig(MODEL, hf_overrides={"text_config": {**SPANS, **text_config}})


def test_a_checkpoint_declares_spans_in_its_text_config():
    config = span_model_config()
    assert config.declares_bidirectional_spans
    assert config.bidirectional_span_layer_types == ("full_attention",)
    assert config.bidirectional_span_max_length == 1024
    assert config.uses_prefix_lm_ranges

    stock = ModelConfig(MODEL)
    assert not stock.declares_bidirectional_spans
    assert not stock.uses_prefix_lm_ranges


def test_multimodal_checkpoints_declare_spans_in_the_text_config():
    """A declaration the text model would not read is refused, not ignored."""
    with pytest.raises(ValueError, match="text config"):
        ModelConfig(MODEL, hf_overrides=SPANS)


def test_architectures_without_spans_refuse_such_checkpoints():
    with pytest.raises(ValueError, match="does not implement bidirectional"):
        ModelConfig("facebook/opt-125m", hf_overrides=SPANS)


@pytest.mark.parametrize(
    "layer_types",
    [["full_attention", "sliding_attention"], ["sliding_attention"]],
    ids=["all", "sliding"],
)
def test_sliding_window_layers_may_apply_spans(layer_types):
    config = span_model_config(bidirectional_span_layer_types=layer_types)
    assert config.bidirectional_span_layer_types == tuple(layer_types)
    assert config.uses_prefix_lm_ranges


def test_other_layer_types_are_refused():
    with pytest.raises(ValueError, match=r"not \['chunked_attention'\]"):
        span_model_config(
            bidirectional_span_layer_types=["full_attention", "chunked_attention"]
        )


@pytest.mark.parametrize("max_length", [None, 0, True, "1024"])
def test_a_valid_max_length_is_required(max_length):
    with pytest.raises(ValueError, match="bidirectional_span_max_length"):
        span_model_config(bidirectional_span_max_length=max_length)


def test_kv_sharing_fast_prefill_is_refused():
    configs = {"cache_config": CacheConfig(kv_sharing_fast_prefill=True)}
    with pytest.raises(ValueError, match="--kv-sharing-fast-prefill"):
        VllmConfig(model_config=span_model_config(), **configs)
    # The same settings serve the checkpoint without the declaration.
    VllmConfig(model_config=ModelConfig(MODEL), **configs)


def test_multi_module_mtp_is_refused():
    """The scheduler shortens its prefill chunks near the prompt end, which can
    end a chunk inside a span."""
    with (
        mock.patch.object(
            VllmConfig,
            "num_prefill_lookahead_tokens",
            new_callable=mock.PropertyMock,
            return_value=2,
        ),
        pytest.raises(ValueError, match="multi-module MTP"),
    ):
        VllmConfig(model_config=span_model_config())


def connector(name: str, *children: str) -> KVTransferConfig:
    extra = {
        "connectors": [{"kv_connector": c, "kv_role": "kv_both"} for c in children]
    }
    return KVTransferConfig(
        kv_connector=name,
        kv_role="kv_both",
        kv_connector_extra_config=extra if name == "MultiConnector" else {},
    )


@pytest.mark.parametrize(
    ("kv_transfer_config", "supported"),
    [
        (connector("OffloadingConnector"), True),
        (connector("SimpleCPUOffloadConnector"), True),
        (connector("ExampleConnector"), False),
        (
            connector(
                "MultiConnector", "OffloadingConnector", "SimpleCPUOffloadConnector"
            ),
            True,
        ),
        (connector("MultiConnector", "OffloadingConnector", "ExampleConnector"), False),
        (connector("MultiConnector"), False),
    ],
    ids=[
        "offloading",
        "simple-cpu",
        "example",
        "multi",
        "multi-example",
        "multi-empty",
    ],
)
def test_only_connectors_keyed_on_block_hashes_serve_spans(
    kv_transfer_config, supported
):
    """Connectors must look external KV up by the block hashes, which carry the
    span keys."""
    model_config = span_model_config()
    if supported:
        VllmConfig(model_config=model_config, kv_transfer_config=kv_transfer_config)
    else:
        with pytest.raises(ValueError, match="supports_bidirectional_spans"):
            VllmConfig(model_config=model_config, kv_transfer_config=kv_transfer_config)
