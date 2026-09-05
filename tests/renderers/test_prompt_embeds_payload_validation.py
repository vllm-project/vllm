# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`prompt_embeds` payloads that are not tensor files must be client errors.

`prompt_embeds` is a base64 blob the caller writes, and every other way of
getting it wrong -- wrong rank, wrong hidden_size, wrong dtype, not a tensor at
all -- already answers 400. A payload that `torch.load` cannot read at all was
the exception: torch has no single error type for that (the zip reader raises
`RuntimeError`, the legacy pickle path raises `UnpicklingError`, `KeyError` or
`IndexError`, an empty payload raises `EOFError`), none of them is a
`ValueError`, and the entrypoints' fallback therefore mapped them to 500.

The assertions go through `create_error_response`, which is what actually turns
an exception into a status code, so they pin the HTTP answer rather than the
exception class alone.
"""

import io
import pickle
import zipfile

import pybase64 as base64
import pytest
import torch

from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import VLLMValidationError
from vllm.renderers.embed_utils import safe_load_prompt_embeds


@pytest.fixture
def model_config():
    from vllm.config import ModelConfig

    return ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="float32",
        seed=0,
        enable_prompt_embeds=True,
    )


def _encode(raw: bytes) -> bytes:
    return base64.b64encode(raw)


def _zip_without_a_tensor() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("junk.txt", "hello")
    return buffer.getvalue()


def _saved_tensor(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


UNPARSEABLE_PAYLOADS = [
    pytest.param(b"", id="empty"),
    pytest.param(b"\x80", id="single-byte"),
    pytest.param(b"\x00\x01\x02\x03not-a-tensor", id="random-bytes"),
    pytest.param(b"hello world this is not a torch file", id="ascii-text"),
    pytest.param(b"PK\x03\x04" + b"\x00" * 20, id="truncated-zip"),
    pytest.param(_zip_without_a_tensor(), id="zip-without-a-tensor"),
    pytest.param(pickle.dumps(42), id="legacy-pickle-of-an-int"),
]


@pytest.mark.parametrize("payload", UNPARSEABLE_PAYLOADS)
def test_unreadable_prompt_embeds_payload_is_a_client_error(model_config, payload):
    with pytest.raises(VLLMValidationError) as excinfo:
        safe_load_prompt_embeds(model_config, _encode(payload))

    assert excinfo.value.parameter == "prompt_embeds"
    assert int(create_error_response(excinfo.value).error.code) == 400


@pytest.mark.parametrize("payload", UNPARSEABLE_PAYLOADS)
def test_the_error_body_stays_bounded(model_config, payload):
    """torch's reason is built from the caller's bytes, so it must be truncated.

    An unpickler that reports the offending global, for instance, quotes a name
    the caller chose; without a bound a large one would be reflected whole.
    """
    padding = b"A" * 100_000
    with pytest.raises(VLLMValidationError) as excinfo:
        safe_load_prompt_embeds(model_config, _encode(payload + padding))

    assert len(str(excinfo.value)) < 500


def test_a_well_formed_payload_still_loads(model_config):
    """Positive control: the healthy path is untouched."""
    hidden_size = model_config.get_hidden_size()
    tensor = torch.zeros(3, hidden_size, dtype=torch.float32)

    loaded = safe_load_prompt_embeds(model_config, _encode(_saved_tensor(tensor)))

    assert loaded.shape == (3, hidden_size)
    assert loaded.dtype == model_config.dtype


def test_a_payload_that_is_not_base64_is_still_a_client_error(model_config):
    """Positive control: the base64 stage already answered 400 and still does."""
    with pytest.raises(ValueError) as excinfo:
        safe_load_prompt_embeds(model_config, b"!!!!not base64!!!!")

    assert int(create_error_response(excinfo.value).error.code) == 400
