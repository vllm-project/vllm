# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import threading
from cProfile import Profile
from dataclasses import replace
from typing import cast

import pytest
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)
from transformers import OPTConfig, TokenizersBackend

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import (
    OpenAIServingCompletion,
    _get_prompt_text_offsets,
    _get_prompt_text_offsets_async,
)
from vllm.inputs import TokensPrompt, tokens_input
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.hf import get_cached_tokenizer, maybe_make_thread_pool

TOKENS = [(1, "abc"), (2, " de"), (3, "!")]
# Text deltas as the detokenizer streams them with stop=["\n\n"]: one char is
# held back until the request finishes, so the text lags the tokens.
TEXT_DELTAS = ["ab", "c d", "e!"]


@pytest.mark.parametrize("reuse_source_offsets", [False, True])
@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("special_count", [1, 2])
@pytest.mark.parametrize(
    "text,source_starts",
    [
        (" É x", [1, 3]),
        (" É x ", [1, 3]),
        ("</s> É x", [0, 5, 7]),
        (" É x </s>", [1, 3, 5]),
    ],
)
def test_normalized_echo_offsets_preserve_special_token_boundaries(
    text, source_starts, add_special_tokens, special_count, reuse_source_offsets
):
    tokenizer = _normalized_tokenizer(special_count)
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    expected = (
        [*[0] * special_count, *source_starts, *[len(text)] * special_count]
        if add_special_tokens
        else source_starts
    )
    prompt_input = tokens_input(ids, prompt=text)
    if reuse_source_offsets:
        prompt_input["prompt_token_offsets"] = tokenizer(
            text, add_special_tokens=add_special_tokens, return_offsets_mapping=True
        )["offset_mapping"]
    offsets = _get_prompt_text_offsets(
        tokenizer, ids, text, prompt_input=prompt_input, source_text=text
    )
    assert offsets == [*expected, len(text)]


@pytest.mark.parametrize(
    "wordpiece_cleanup,pooled,normalize",
    [
        (None, False, True),
        (False, True, True),
        (False, False, False),
        (True, False, True),
    ],
)
def test_echo_offsets_reuse_source_before_singleton_decodes(
    wordpiece_cleanup, pooled, normalize
):
    num_tokens = 8192
    backend = Tokenizer(
        models.WordLevel({"x": 0, "é": 1, "[UNK]": 2}, unk_token="[UNK]")
    )
    if normalize:
        backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = (
        decoders.Fuse()
        if wordpiece_cleanup is None
        else decoders.WordPiece(cleanup=wordpiece_cleanup)
    )
    tokenizer = TokenizersBackend(tokenizer_object=backend)
    tokenizer = get_cached_tokenizer(tokenizer)
    if pooled:
        tokenizer = maybe_make_thread_pool(tokenizer)
    text = "x" * (num_tokens - 1) + ("É" if normalize else "𐍈")
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    assert len(ids) == num_tokens
    prompt_input = tokens_input(ids, prompt=text)
    prompt_input["prompt_token_offsets"] = encoding["offset_mapping"]
    with Profile() as profile:
        offsets = _get_prompt_text_offsets(
            tokenizer, ids, text, prompt_input=prompt_input, source_text=text
        )
    assert offsets == list(range(num_tokens + 1))
    assert not any(
        entry.code
        in (tokenizer.decode.__func__.__code__, tokenizer._decode.__func__.__code__)
        for entry in profile.getstats()
    )
    assert not any(
        isinstance(entry.code, str) and "'id_to_token'" in entry.code
        for entry in profile.getstats()
    )
    reuse_source = wordpiece_cleanup is not True
    assert sum(
        entry.callcount
        for entry in profile.getstats()
        if isinstance(entry.code, str) and "'decode_batch'" in entry.code
    ) == int(not reuse_source)


@pytest.mark.parametrize(
    "text,options,retain",
    [
        ("É x", {"echo": True, "logprobs": 0}, True),
        ("É", {"echo": True}, False),
        ("É", {"logprobs": 0}, False),
        ("É", {"echo": True, "logprobs": 0, "return_token_ids": True}, False),
        ("É", {"echo": True, "logprobs": 0, "return_tokens_as_token_ids": True}, False),
        ("x", {"echo": True, "logprobs": 0}, False),
    ],
    ids=["normalized", "no-logprobs", "no-echo", "token-ids", "placeholders", "ascii"],
)
def test_echo_tokenization_retains_source_offsets(tmp_path, text, options, retain):
    OPTConfig(vocab_size=5).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), dtype="float32", max_model_len=32)
    )
    tokenizer = get_cached_tokenizer(_normalized_tokenizer())
    renderer = HfRenderer(config, tokenizer)
    request = CompletionRequest(
        model="m", prompt=text, add_special_tokens=False, **options
    )
    try:
        params = request.build_tok_params(config.model_config)
        reference = renderer.tokenize_prompt(
            {"prompt": text}, replace(params, retain_token_offsets=False)
        )
        prompt = cast(TokensPrompt, renderer.tokenize_prompt({"prompt": text}, params))
        assert {
            key: value for key, value in prompt.items() if key != "prompt_token_offsets"
        } == reference
        assert ("prompt_token_offsets" in prompt) == retain
        if retain:
            assert prompt["prompt_token_offsets"] == [(0, 1), (2, 3)]
    finally:
        renderer.shutdown()


@pytest.mark.parametrize("wordpiece", [False, True])
def test_echo_offsets_keep_literal_boundaries_with_untrimmed_source(wordpiece):
    backend = Tokenizer(models.WordLevel({"é": 0, "x": 1}))
    # Even an unchanged replacement can collapse source spans.
    backend.normalizer = normalizers.Sequence(
        [normalizers.Replace("éx", "éx"), normalizers.Lowercase()]
    )
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = (
        decoders.WordPiece(cleanup=False) if wordpiece else decoders.Fuse()
    )
    tokenizer = TokenizersBackend(tokenizer_object=backend)
    text = "éx"
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert encoding["offset_mapping"] == [(1, 2), (1, 2)]
    prompt_input = tokens_input(encoding["input_ids"], prompt=text)
    prompt_input["prompt_token_offsets"] = encoding["offset_mapping"]
    assert _get_prompt_text_offsets(
        tokenizer,
        encoding["input_ids"],
        text,
        prompt_input=prompt_input,
        source_text=text,
    ) == list(range(len(text) + 1))


def test_echo_offsets_keep_literal_boundaries_with_wordpiece_cleanup():
    backend = Tokenizer(models.WordLevel({"do not": 0, "É": 1}))
    backend.normalizer = normalizers.Sequence(
        [normalizers.Replace("do not", "do not"), normalizers.Lowercase()]
    )
    backend.decoder = decoders.WordPiece()
    tokenizer = TokenizersBackend(tokenizer_object=backend)
    tokenizer.add_tokens([AddedToken("É", normalized=False)])
    text = "Édo not"
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    assert encoding["offset_mapping"] == [(0, 1), (6, 7)]
    assert tokenizer.decode(ids) == "É don't"
    assert "".join(tokenizer.decode([token_id]) for token_id in ids) == text
    prompt_input = tokens_input(ids, prompt=text)
    prompt_input["prompt_token_offsets"] = encoding["offset_mapping"]
    assert _get_prompt_text_offsets(
        tokenizer, ids, text, prompt_input=prompt_input, source_text=text
    ) == [0, 1, len(text)]


@pytest.mark.parametrize("split_unicode", [False, True])
@pytest.mark.parametrize("reuse_source_offsets", [False, True])
@pytest.mark.parametrize("text", ["é x", "É x"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("trim", [False, True])
def test_literal_echo_offsets_precede_trimmed_source_offsets(
    split_unicode, reuse_source_offsets, text, nested, trim
):
    backend = Tokenizer(
        models.BPE(
            {"Ã": 0, "©": 1, "Ã©": 2, "Ġ": 3, "x": 4, "Ġx": 5},
            ([] if split_unicode else [("Ã", "©")]) + [("Ġ", "x")],
        )
    )
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    processor = processors.ByteLevel(trim_offsets=trim)
    backend.post_processor = processors.Sequence([processor]) if nested else processor
    backend.decoder = decoders.Sequence(
        [decoders.ByteLevel(), decoders.Replace("y", "Y")]
    )
    tokenizer = TokenizersBackend(tokenizer_object=backend)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    assert ids == ([0, 1, 5] if split_unicode else [2, 5])
    assert encoding["offset_mapping"][-1] == (2 if trim else 1, 3)
    prompt_input = tokens_input(ids, prompt=text)
    if reuse_source_offsets:
        prompt_input["prompt_token_offsets"] = encoding["offset_mapping"]
    offsets = _get_prompt_text_offsets(
        tokenizer, ids, text, prompt_input=prompt_input, source_text=text
    )
    # Unsupported contextual decoding must not replace legacy offsets with
    # source spans that trimmed a token's leading space.
    expected = (
        None
        if trim and (split_unicode or text == "É x")
        else [0, *([0] if split_unicode else []), 1, len(text)]
    )
    assert offsets == expected


def _normalized_tokenizer(special_count=1):
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "<s>": 1, "</s>": 2, "é": 3, "x": 4}, unk_token="[UNK]"
        )
    )
    backend.normalizer = normalizers.Sequence(
        [normalizers.Strip(), normalizers.Lowercase()]
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.WordPiece(cleanup=False)
    backend.post_processor = processors.TemplateProcessing(
        single=" ".join([*["<s>"] * special_count, "$A", *["</s>"] * special_count]),
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="[UNK]"
    )
    return tokenizer


@pytest.mark.parametrize("stale_cache", ["text", "ids", "length", "bounds"])
def test_echo_offsets_ignore_stale_source_offsets(stale_cache):
    tokenizer = _normalized_tokenizer()
    text = " É x"
    ids = tokenizer.encode(text, add_special_tokens=False)
    prompt_input = tokens_input(ids, prompt=text)
    prompt_input["prompt_token_offsets"] = [(1, 2), (3, 4)]
    source_text = text
    if stale_cache == "text":
        source_text = "É x"
        prompt_input["prompt_token_offsets"] = [(0, 1), (2, 3)]
    elif stale_cache == "ids":
        prompt_input["prompt_token_ids"] = list(reversed(ids))
        prompt_input["prompt_token_offsets"] = [(0, 1), (1, 2)]
    elif stale_cache == "length":
        prompt_input["prompt_token_offsets"] = [(1, 2)]
    else:
        prompt_input["prompt_token_offsets"] = [(1, 2), (99, 100)]
    assert _get_prompt_text_offsets(
        tokenizer, ids, text, prompt_input=prompt_input, source_text=source_text
    ) == [1, 3, len(text)]


@pytest.mark.parametrize(
    "echo_mask", [[False], [True, True, True, False], [False, True, True, False]]
)
def test_echo_offsets_reuse_source_offsets_after_masking(echo_mask):
    tokenizer = _normalized_tokenizer()
    text = " É x"
    encoding = tokenizer(text, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    prompt_input = tokens_input(ids, prompt=text)
    prompt_input["prompt_token_offsets"] = encoding["offset_mapping"]
    prompt_input["prompt_echo_mask"] = echo_mask
    expected = [
        offset
        for i, offset in enumerate([0, 1, 3, len(text)])
        if i >= len(echo_mask) or echo_mask[i]
    ]
    assert _get_prompt_text_offsets(
        tokenizer, ids, text, prompt_input=prompt_input, source_text=text
    ) == [*expected, len(text)]


@pytest.mark.asyncio
async def test_echo_offsets_use_renderer_tokenizer_executor(tmp_path):
    OPTConfig(vocab_size=5).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="float32",
            max_model_len=32,
            renderer_num_workers=1,
        )
    )
    renderer = HfRenderer(config, _normalized_tokenizer())
    text = " É x"
    ids = renderer.get_tokenizer().encode(text, add_special_tokens=False)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    def occupy_worker():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=10)

    worker = renderer._executor.submit(occupy_worker)
    task = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task = asyncio.create_task(
            _get_prompt_text_offsets_async(
                renderer,
                ids,
                text,
                prompt_input=tokens_input(ids, prompt=text),
                source_text=text,
            )
        )
        checkpoint = asyncio.Event()
        loop.call_soon(checkpoint.set)
        await checkpoint.wait()
        # The event loop progressed, but alignment waits for the renderer pool.
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, timeout=5) == [1, 3, len(text)]
        await asyncio.wrap_future(worker)
    finally:
        release.set()
        if task is not None:
            await task
        renderer.shutdown()

    # The closed renderer pool must reject work, even if the default pool is open.
    with pytest.raises(
        RuntimeError, match="cannot schedule new futures after shutdown"
    ):
        await _get_prompt_text_offsets_async(
            renderer,
            ids,
            text,
            prompt_input=tokens_input(ids, prompt=text),
            source_text=text,
        )


def _logprobs(i: int) -> dict[int, Logprob]:
    token_id, token = TOKENS[i]
    return {token_id: Logprob(logprob=-1.0, rank=1, decoded_token=token)}


def _output(text: str, idxs: list[int], finished: bool) -> RequestOutput:
    return RequestOutput(
        request_id="r",
        prompt="Hi",
        prompt_token_ids=[9],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[TOKENS[i][0] for i in idxs],
                cumulative_logprob=None,
                logprobs=[_logprobs(i) for i in idxs],
                finish_reason="stop" if finished else None,
            )
        ],
        finished=finished,
    )


def _serving() -> OpenAIServingCompletion:
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    serving.enable_log_outputs = False
    serving.enable_per_request_metrics = False
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False
    return serving


def _request(stream: bool) -> CompletionRequest:
    return CompletionRequest(
        model="m", prompt="Hi", max_tokens=8, logprobs=0, stop=["\n\n"], stream=stream
    )


@pytest.mark.asyncio
async def test_stream_text_offset_matches_non_stream_with_stop_holdback():
    async def results():
        for i, text in enumerate(TEXT_DELTAS):
            yield 0, _output(text, [i], finished=i == len(TEXT_DELTAS) - 1)

    serving = _serving()
    stream_offsets = []
    async for line in serving.completion_stream_generator(
        _request(stream=True),
        [tokens_input([9], prompt="Hi")],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            continue
        for choice in json.loads(data)["choices"]:
            stream_offsets += choice["logprobs"]["text_offset"]

    response = serving.request_output_to_completion_response(
        [_output("".join(TEXT_DELTAS), [0, 1, 2], finished=True)],
        _request(stream=False),
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )
    assert response.choices[0].logprobs is not None
    assert response.choices[0].logprobs.text_offset == [0, 3, 6]
    assert stream_offsets == [0, 3, 6]


def test_non_stream_response_omits_null_prompt_tokens_details():
    """The OpenAI spec types `usage.prompt_tokens_details` as an object, not
    nullable, so it is omitted when `--enable-prompt-tokens-details` is off."""
    response = _serving().request_output_to_completion_response(
        [_output("abc", [0], finished=True)],
        _request(stream=False),
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )

    assert "prompt_tokens_details" not in response.model_dump()["usage"]


PROMPT_IDS = [9, 1, 2, 3]


def _aborted_output(prompt_logprobs) -> RequestOutput:
    """A request aborted mid-prefill: prompt logprobs cover only the processed
    prompt tokens, or none of them, and nothing was generated."""
    return RequestOutput(
        request_id="r",
        prompt="Hi",
        prompt_token_ids=PROMPT_IDS,
        prompt_logprobs=prompt_logprobs,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=[],
                finish_reason="abort",
            )
        ],
        finished=True,
    )


def _echo_request(stream: bool) -> CompletionRequest:
    return CompletionRequest(
        model="m",
        prompt=PROMPT_IDS,
        max_tokens=8,
        logprobs=0,
        echo=True,
        stream=stream,
        return_tokens_as_token_ids=True,
    )


@pytest.mark.parametrize("prompt_logprobs", [[None], [None, _logprobs(0)], None])
def test_echo_logprobs_of_request_aborted_mid_prefill(prompt_logprobs):
    response = _serving().request_output_to_completion_response(
        [_aborted_output(prompt_logprobs)],
        _echo_request(stream=False),
        "r",
        0,
        "m",
        None,
        RequestResponseMetadata(request_id="r"),
    )

    choice = response.choices[0]
    assert choice.finish_reason == "abort"
    assert choice.logprobs is not None
    assert len(choice.logprobs.tokens) == len(PROMPT_IDS)
    # Prompt positions the engine never reached have no logprob
    assert choice.logprobs.token_logprobs[2:] == [None, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_logprobs", [[None], [None, _logprobs(0)], None])
async def test_stream_echo_logprobs_of_request_aborted_mid_prefill(prompt_logprobs):
    async def results():
        yield 0, _aborted_output(prompt_logprobs)

    tokens = []
    async for line in _serving().completion_stream_generator(
        _echo_request(stream=True),
        [tokens_input(PROMPT_IDS, prompt="Hi")],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        assert '"error"' not in data
        if data == "[DONE]":
            continue
        for choice in json.loads(data)["choices"]:
            if choice["logprobs"] is not None:
                tokens += choice["logprobs"]["tokens"]

    assert len(tokens) == len(PROMPT_IDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [0, 8])
async def test_stream_echo_without_logprobs_skips_prompt_padding(
    monkeypatch, max_tokens
):
    def pad(*args):
        raise AssertionError("prompt logprobs padded without logprobs requested")

    monkeypatch.setattr(
        "vllm.entrypoints.openai.completion.serving._padded_prompt_logprobs", pad
    )
    request = _echo_request(stream=True)
    request.logprobs = None
    request.max_tokens = max_tokens

    async def results():
        yield 0, _aborted_output(None)

    async for line in _serving().completion_stream_generator(
        request,
        [tokens_input(PROMPT_IDS, prompt="Hi")],
        results(),
        "r",
        0,
        "m",
        num_prompts=1,
        tokenizer=None,
        request_metadata=RequestResponseMetadata(request_id="r"),
    ):
        data = line.removeprefix("data: ").strip()
        assert '"error"' not in data
        if data != "[DONE]":
            assert all(c["logprobs"] is None for c in json.loads(data)["choices"])
