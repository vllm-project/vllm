# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import pytest_asyncio
from tokenizers import Tokenizer, decoders, processors
from transformers import TokenizersBackend

from tests.utils import RemoteOpenAIServer
from vllm.tokenizers import get_tokenizer

UNICODE_SERVER_ARGS = [
    "--dtype",
    "float16",
    "--max-model-len",
    "128",
    "--max-num-seqs",
    "4",
    "--attention-backend",
    "TRITON_ATTN",
    "--gpu-memory-utilization",
    "0.15",
    "--kv-cache-memory-bytes",
    "67108864",
]


@pytest.fixture(scope="module")
def server(request):
    model_name, extra_args = request.param
    if isinstance(extra_args, str):
        extra_args = request.getfixturevalue(extra_args)
    with RemoteOpenAIServer(
        model_name,
        ["--enforce-eager", "--no-enable-prefix-caching", *extra_args],
        env_dict={"VLLM_USE_RUST_FRONTEND": "0", "VLLM_BATCH_INVARIANT": "1"},
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def _get_echo_logprobs(client, model, prompt, stream, max_tokens, extra_body):
    reference = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max(1, max_tokens),
        temperature=0,
        logprobs=0,
        extra_body={**extra_body, "return_token_ids": True},
    )
    response = await client.completions.create(
        model=model,
        prompt=prompt,
        echo=True,
        max_tokens=max_tokens,
        temperature=0,
        logprobs=0,
        stream=stream,
        extra_body=extra_body,
    )
    choices = (
        [chunk.choices[0] async for chunk in response] if stream else response.choices
    )
    text = "".join(c.text for c in choices)
    offsets = [o for c in choices for o in c.logprobs.text_offset]
    tokens = [t for c in choices for t in c.logprobs.tokens]
    return reference.choices[0], text, offsets, tokens


def _unicode_trimmed_server_args(tmp_path_factory, *, replace=False):
    tokenizer = cast(TokenizersBackend, get_tokenizer("facebook/opt-125m"))
    backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    backend.post_processor = processors.ByteLevel(trim_offsets=True)
    steps = [backend.decoder, decoders.Fuse()]
    if replace:
        steps.append(decoders.Replace("\r\n", "\n"))
    backend.decoder = decoders.Sequence(steps)
    path = tmp_path_factory.mktemp("unicode-trimmed-tokenizer")
    TokenizersBackend(tokenizer_object=backend).save_pretrained(path)
    return [*UNICODE_SERVER_ARGS, "--tokenizer", str(path)]


@pytest.fixture(scope="module")
def unicode_trimmed_server_args(tmp_path_factory):
    return _unicode_trimmed_server_args(tmp_path_factory)


@pytest.fixture(scope="module")
def unicode_unknown_trimmed_server_args(tmp_path_factory):
    return _unicode_trimmed_server_args(tmp_path_factory, replace=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server,contextual",
    [
        (("facebook/opt-125m", "unicode_trimmed_server_args"), True),
        (("facebook/opt-125m", "unicode_unknown_trimmed_server_args"), False),
    ],
    indirect=["server"],
    scope="module",
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 3])
@pytest.mark.parametrize("return_token_offsets", [False, True])
@pytest.mark.parametrize(
    "prompt,expected",
    [("é x", [0, 1]), ("é  x", [0, 1, 2]), ("𐍈 x", [0, 0, 0, 0, 1])],
)
async def test_echo_unicode_offsets_preserve_trimmed_spaces(
    client,
    server,
    contextual,
    stream,
    max_tokens,
    return_token_offsets,
    prompt,
    expected,
):
    model = (await client.models.list()).data[0].id
    reference, text, offsets, tokens = await _get_echo_logprobs(
        client,
        model,
        prompt,
        stream,
        max_tokens,
        {"add_special_tokens": False, "return_token_offsets": return_token_offsets},
    )
    count = len(expected)
    if not contextual and prompt == "𐍈 x":
        # Keep legacy split-character offsets for unsupported decoders, but
        # preserve the space token's start rather than its trimmed source span.
        expected = [0, 1, 1, 1, 1]
    assert len(reference.prompt_token_ids) == count
    assert text == prompt + (reference.text if max_tokens else "")
    assert offsets[:count] == expected
    assert tokens[count - 1] == " x"
    if len(set(expected)) == count:
        # Split UTF-8 tokens cannot reconstruct the prompt independently.
        assert "".join(tokens[:count]) == prompt
    assert offsets[count:] == (
        [len(prompt) + o for o in reference.logprobs.text_offset] if max_tokens else []
    )


@pytest.fixture(scope="module")
def unicode_eos_server_args(tmp_path_factory):
    tokenizer = cast(
        TokenizersBackend, get_tokenizer("mistralai/Mistral-7B-Instruct-v0.3")
    )
    backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.bos_token_id),
            ("</s>", tokenizer.eos_token_id),
        ],
    )
    path = tmp_path_factory.mktemp("unicode-eos-tokenizer")
    TokenizersBackend(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>"
    ).save_pretrained(path)
    return [*UNICODE_SERVER_ARGS, "--tokenizer", str(path)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        (
            "mistralai/Mistral-7B-Instruct-v0.3",
            "unicode_eos_server_args",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 3])
@pytest.mark.parametrize("prompt", [" 𐍈x", " 𐍈x </s>", " 𐍈x  "])
async def test_echo_unicode_offsets_with_added_eos(
    client, server, stream, max_tokens, prompt
):
    model = (await client.models.list()).data[0].id
    reference, text, offsets, tokens = await _get_echo_logprobs(
        client, model, prompt, stream, max_tokens, {}
    )
    count = len(reference.prompt_token_ids)
    assert text == prompt + (reference.text if max_tokens else "")
    assert tokens[0] == "<s>" and offsets[0] == 0
    assert tokens[count - 1] == "</s>" and offsets[count - 1] == len(prompt)
    assert offsets[:count] == sorted(offsets[:count])
    if "</s>" in prompt:
        assert tokens[count - 2] == "</s>"
        assert offsets[count - 2] == prompt.index("</s>")
    if max_tokens:
        assert offsets[count:] == [
            len(prompt) + o for o in reference.logprobs.text_offset
        ]
    else:
        assert len(offsets) == count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        (
            model,
            [
                *UNICODE_SERVER_ARGS,
                "--tokenizer-mode",
                mode,
            ],
        )
        for model, mode in [
            ("facebook/opt-125m", "auto"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "auto"),
            ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        ]
    ],
    indirect=True,
    ids=["bytelevel", "hf-fallback", "native-fallback"],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_tokens", [0, 3])
@pytest.mark.parametrize("prompt", ["𐍈x", " 𐍈x", "a�x", "a�x suffix"])
@pytest.mark.parametrize("add_special_tokens", [False, True])
async def test_echo_unicode_offsets(
    client, server, stream, max_tokens, prompt, add_special_tokens
):
    """Prompt offsets refer to echoed characters, including split UTF-8."""
    model = (await client.models.list()).data[0].id
    extra_body = {"add_special_tokens": add_special_tokens}
    truncated = prompt.endswith(" suffix")
    if truncated:
        # Retain the complete Unicode prefix, including any added BOS in the
        # model inputs. The echo must exclude that added BOS after truncation.
        prefix = prompt.removesuffix(" suffix")
        prefix_response = await client.completions.create(
            model=model,
            prompt=prefix,
            max_tokens=1,
            temperature=0,
            extra_body={**extra_body, "return_token_ids": True},
        )
        extra_body.update(
            truncate_prompt_tokens=len(prefix_response.choices[0].prompt_token_ids),
            truncation_side="right",
        )
    reference, text, offsets, tokens = await _get_echo_logprobs(
        client, model, prompt, stream, max_tokens, extra_body
    )
    prompt_ids = reference.prompt_token_ids
    assert prompt_ids is not None
    if truncated:
        assert prompt_ids == prefix_response.choices[0].prompt_token_ids
        prompt = prefix
    count = len(prompt_ids) - int(truncated and add_special_tokens)
    assert text == prompt + (reference.text if max_tokens else "")
    assert offsets[count - 1] == len(prompt) - 1
    assert tokens[count - 1] == "x"
    if prompt == "𐍈x":
        assert offsets[: count - 1] == [0] * (count - 1)
    elif prompt == "a�x":
        assert offsets[:count] == [0] * (count - 2) + [1, 2]
    if max_tokens:
        assert offsets[count:] == [
            len(prompt) + o for o in reference.logprobs.text_offset
        ]
        assert tokens[count:] == reference.logprobs.tokens
    else:
        assert len(offsets) == count
