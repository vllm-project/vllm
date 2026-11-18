# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import torch

from vllm.entrypoints.pooling.embed.protocol import CohereEmbedRequest
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.pooling.protocol import IOProcessorResponse
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.pooling.utils import (
    encode_pooling_output_float_or_ndarray,
    merge_chat_template_kwargs,
)
from vllm.entrypoints.serve.utils.api_utils import create_json_response
from vllm.exceptions import VLLMValidationError
from vllm.outputs import PoolingOutput, PoolingRequestOutput

pytestmark = pytest.mark.skip_global_cleanup


def _pooling_output(data):
    return SimpleNamespace(outputs=SimpleNamespace(data=data))


def test_encode_pooling_output_float_or_ndarray_returns_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    encoded = encode_pooling_output_float_or_ndarray(output)

    assert isinstance(encoded, np.ndarray)
    np.testing.assert_allclose(encoded, [1.0, 2.0, 3.0])


def test_json_response_serializes_numpy_array():
    output = _pooling_output(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    encoded = encode_pooling_output_float_or_ndarray(output)
    response = create_json_response({"embedding": encoded})
    assert json.loads(response.body)["embedding"] == pytest.approx([1.0, 2.0, 3.0])


def test_encode_pooling_output_float_or_ndarray_falls_back_to_list():
    class DataWithUnsupportedNumpy:
        def is_contiguous(self):
            return True

        def numpy(self):
            raise TypeError("unsupported dtype")

        def tolist(self):
            return [1.0, 2.0, 3.0]

    output = _pooling_output(DataWithUnsupportedNumpy())

    assert encode_pooling_output_float_or_ndarray(output) == [1.0, 2.0, 3.0]


@pytest.fixture
def pooling_outputs():
    return [
        PoolingRequestOutput(
            request_id="req",
            outputs=PoolingOutput(data=torch.tensor([1.25, -2.5, 0.0, 3.0])),
            prompt_token_ids=[1, 2],
            num_cached_tokens=0,
            finished=True,
        )
    ]


@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@pytest.mark.parametrize("api", ["embedding", "pooling"])
def test_pooling_json_response_preserves_encoding(
    pooling_outputs, api, encoding_format
):
    if api == "embedding":
        build_response = object.__new__(ServingEmbedding)._openai_json_response
        data_key = "embedding"
    else:
        build_response = object.__new__(
            ServingPooling
        ).request_output_to_pooling_json_response
        data_key = "data"
    response = build_response(
        pooling_outputs, "req", 123, "model", encoding_format, "float32", "little"
    )
    body = json.loads(response.body)
    assert response.status_code == 200
    assert response.media_type == "application/json"
    assert body["id"] == "req"
    assert body["created"] == 123
    assert body["model"] == "model"
    assert body["usage"]["prompt_tokens"] == body["usage"]["total_tokens"] == 2
    encoded = body["data"][0][data_key]
    if encoding_format == "base64":
        encoded = np.frombuffer(pybase64.b64decode(encoded), dtype="<f4")
    np.testing.assert_array_equal(encoded, [1.25, -2.5, 0.0, 3.0])


def test_cohere_json_response_preserves_omitted_fields(pooling_outputs):
    serving = object.__new__(ServingEmbedding)
    ctx = SimpleNamespace(
        request=CohereEmbedRequest(texts=["hello"]),
        final_res_batch=pooling_outputs,
        request_id="req",
    )
    body = json.loads(serving._build_cohere_response_from_ctx(ctx).body)
    assert body["embeddings"] == {"float": [[1.25, -2.5, 0.0, 3.0]]}
    assert body["texts"] == ["hello"]
    assert body["meta"]["billed_units"]["input_tokens"] == 2


def test_pooling_plugin_json_response_preserves_payload():
    serving = object.__new__(ServingPooling)
    ctx = SimpleNamespace(
        response=IOProcessorResponse(
            request_id="req", created_at=123, data={"text": "中文", "scores": [1, 2]}
        )
    )
    body = json.loads(serving._build_response(ctx).body)
    assert body == ctx.response.model_dump()


class TestMergeTemplateKwargs:
    """Unit tests for `pooling.utils.merge_chat_template_kwargs`."""

    # --- Flags are always materialized ------------------------------------

    def test_empty_kwargs_materializes_both_false(self):
        """Regression: offline defaults are ``False`` and must be written.

        A renderer that defaults the key to ``True`` must see the explicit
        ``False`` instead of a missing key.
        """
        assert merge_chat_template_kwargs(
            None,
            add_generation_prompt=False,
            continue_final_message=False,
        ) == {
            "add_generation_prompt": False,
            "continue_final_message": False,
        }

    def test_existing_kwargs_are_preserved_and_flags_appended(self):
        assert merge_chat_template_kwargs(
            {"existing": 1},
            add_generation_prompt=False,
            continue_final_message=False,
        ) == {
            "existing": 1,
            "add_generation_prompt": False,
            "continue_final_message": False,
        }

    def test_true_flags_are_materialized(self):
        assert merge_chat_template_kwargs(
            None,
            add_generation_prompt=True,
            continue_final_message=False,
        ) == {
            "add_generation_prompt": True,
            "continue_final_message": False,
        }

    # --- Precedence: existing kwargs win over top-level flags --------------

    def test_existing_kwargs_win_over_top_level_true(self):
        """An explicit ``False`` in kwargs must survive a top-level ``True``."""
        assert merge_chat_template_kwargs(
            {"add_generation_prompt": False, "continue_final_message": False},
            add_generation_prompt=True,
            continue_final_message=True,
        ) == {
            "add_generation_prompt": False,
            "continue_final_message": False,
        }

    # --- Conflict detection ------------------------------------------------

    def test_rejects_mutually_exclusive_top_level_flags(self):
        with pytest.raises(VLLMValidationError, match="Cannot set both"):
            merge_chat_template_kwargs(
                None,
                add_generation_prompt=True,
                continue_final_message=True,
            )

    @pytest.mark.parametrize(
        ("kwargs", "add_generation_prompt", "continue_final_message"),
        [
            ({"continue_final_message": True}, True, False),
            ({"add_generation_prompt": True}, False, True),
        ],
        ids=["top_level_add_vs_kwargs_continue", "top_level_continue_vs_kwargs_add"],
    )
    def test_rejects_mutually_exclusive_flags_across_layers(
        self, kwargs, add_generation_prompt, continue_final_message
    ):
        """`__post_init__` only sees the top-level fields, so a cross-layer
        conflict has to be caught here."""
        with pytest.raises(VLLMValidationError, match="Cannot set both"):
            merge_chat_template_kwargs(
                kwargs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )

    def test_explicit_false_does_not_trigger_conflict(self):
        """A ``False`` flag in kwargs must not conflict with a top-level
        ``True`` on the other flag."""
        assert merge_chat_template_kwargs(
            {"continue_final_message": False},
            add_generation_prompt=True,
            continue_final_message=False,
        ) == {
            "continue_final_message": False,
            "add_generation_prompt": True,
        }
