# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import threading

import pytest

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import (
    OpenAIServingCompletion,
    _get_prompt_text_offsets,
    _get_prompt_text_offsets_async,
)
from vllm.inputs import tokens_input
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput

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
    assert _get_prompt_text_offsets(
        tokenizer, ids, text, prompt_input=prompt_input, source_text=text
    ) == [*expected, len(text)]


def _normalized_tokenizer(special_count=1):
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
    )
    from transformers import TokenizersBackend

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
    from transformers import OPTConfig

    from vllm.config import ModelConfig, VllmConfig
    from vllm.renderers.hf import HfRenderer

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
