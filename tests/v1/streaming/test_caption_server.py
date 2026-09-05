# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-GPU tests for the REST streaming server plumbing.

A fake EngineClient consumes the StreamingInput async-generator and emits canned
DELTA outputs with finish_reason transitions, so the session/serving layer and
the chunk-builder are exercised without a model or GPU.
"""

import asyncio
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm.entrypoints.openai.streaming import chunking
from vllm.entrypoints.openai.streaming.protocol import (
    SamplingConfig,
    SessionRequest,
)
from vllm.entrypoints.openai.streaming.serving import (
    OpenAIServingStreaming,
    StreamingError,
    _build_structured_outputs,
)
from vllm.sampling_params import RequestOutputKind
from vllm.v1.streaming.retention import StreamingRetentionParams


def _serving(engine, *, sampling=None):
    # Retention is per-session now (supplied in SessionRequest); the serving
    # object only carries the default + sampling config.
    return OpenAIServingStreaming(engine, sampling=sampling)


def _png_bytes(w=16, h=16, color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Fake engine
# --------------------------------------------------------------------------- #


def _completion(text, token_ids, finish_reason):
    return SimpleNamespace(text=text, token_ids=token_ids, finish_reason=finish_reason)


def _output(completions, finished):
    return SimpleNamespace(
        outputs=completions, finished=finished, prompt_token_ids=None
    )


class FakeRenderer:
    """Minimal stand-in for the model renderer: flattens messages to a string
    and collects image_pil parts into multi_modal_data (no real chat template)."""

    async def render_messages_async(self, messages, params):
        parts: list[str] = []
        images: list = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                parts.append(f"<|{msg['role']}|>{content}")
                continue
            seg = [f"<|{msg['role']}|>"]
            for part in content:
                if part["type"] == "image_pil":
                    seg.append("<image>")
                    images.append(part["image_pil"])
                elif part["type"] == "text":
                    seg.append(part["text"])
            parts.append("".join(seg))
        prompt = {"prompt": "".join(parts), "multi_modal_data": {"image": images}}
        return [], prompt

    async def render_chat_async(
        self, conversations, chat_params, tok_params=None, **kwargs
    ):
        rendered = [
            await self.render_messages_async(m, chat_params) for m in conversations
        ]
        return [c for c, _ in rendered], [p for _, p in rendered]


class FakeEngine:
    """Emits two deltas + a stop per pushed frame; a finished output at end."""

    def __init__(self, video_limit=64):
        self.model_config = SimpleNamespace(
            model="fake-model",
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda m: video_limit
            ),
        )
        self.renderer = FakeRenderer()
        self.seen_chunks = []
        self.aborted = []

    async def generate(self, prompt, sampling_params, request_id):
        idx = 0
        async for chunk in prompt:  # one StreamingInput per pushed frame
            self.seen_chunks.append(chunk)
            yield _output([_completion(f"frame{idx}", [1, 2], None)], finished=False)
            yield _output([_completion("!", [3], "stop")], finished=False)
            idx += 1
        yield _output([_completion("", [], "stop")], finished=True)

    async def abort(self, request_id):
        self.aborted.append(request_id)


async def _drive(serving, req, frames):
    sess = await serving.create_session(req)
    out = []
    for _ in range(frames):
        out.append(await serving.push_frame(sess.session_id, _png_bytes()))
    close = await serving.close_session(sess.session_id)
    return sess, out, close


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def test_decode_frame():
    img = chunking.decode_frame(_png_bytes(20, 12))
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB" and img.size == (20, 12)  # (width, height)


def test_build_messages_first_vs_later():
    frame = chunking.decode_frame(_png_bytes())
    first = chunking.build_messages(frame, "SYS-PROMPT", "Q?", is_first=True)
    later = chunking.build_messages(frame, "SYS-PROMPT", "Q?", is_first=False)
    # First turn opens with system + an image+question user turn.
    assert first[0] == {"role": "system", "content": "SYS-PROMPT"}
    first_user = first[1]["content"]
    assert any(p["type"] == "image_pil" for p in first_user)
    assert any(p["type"] == "text" and p["text"] == "Q?" for p in first_user)
    # Later turns are image-only: no system, no repeated question.
    assert len(later) == 1 and later[0]["role"] == "user"
    assert [p["type"] for p in later[0]["content"]] == ["image_pil"]


def test_build_chunk_renders_via_renderer():
    async def main():
        from vllm.sampling_params import SamplingParams

        frame = chunking.decode_frame(_png_bytes())
        sp = SamplingParams(max_tokens=8, output_kind=RequestOutputKind.DELTA)
        r = FakeRenderer()
        first = await chunking.build_chunk(r, frame, True, "Q?", "SYS-PROMPT", sp)
        later = await chunking.build_chunk(r, frame, False, "Q?", "SYS-PROMPT", sp)
        assert "SYS-PROMPT" in first.prompt["prompt"]
        assert "SYS-PROMPT" not in later.prompt["prompt"]
        assert first.prompt["multi_modal_data"]["image"]  # frame attached

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# structured outputs + sampling params
# --------------------------------------------------------------------------- #


def test_structured_outputs_choice_and_mutual_exclusion():
    so = _build_structured_outputs(SamplingConfig(guided_choice=["0", "1", "2"]))
    assert so is not None and so.choice == ["0", "1", "2"]
    assert _build_structured_outputs(SamplingConfig()) is None
    with pytest.raises(ValueError):  # launch-config validation (not a client 4xx)
        _build_structured_outputs(SamplingConfig(guided_choice=["a"], guided_regex="x"))


def test_sampling_params_carry_retention_and_delta():
    serving = _serving(FakeEngine())
    retention = StreamingRetentionParams(max_video_segments=6)
    sp = serving._build_sampling_params(retention, SamplingConfig())
    assert sp.output_kind is RequestOutputKind.DELTA
    ret = sp.extra_args["streaming_retention"]
    assert isinstance(ret, StreamingRetentionParams) and ret.max_video_segments == 6


def test_config_endpoint_exposes_defaults():
    # Configure a non-default sampling default so the round-trip is real, not
    # a comparison of two identical default constructors.
    serving = _serving(FakeEngine(), sampling=SamplingConfig(max_tokens=42))
    cfg = serving.config()
    assert cfg.model == "fake-model"
    assert cfg.sampling.max_tokens == 42
    # Retention is per-session; the endpoint exposes the dataclass default.
    assert (
        cfg.retention.max_video_segments
        == StreamingRetentionParams().max_video_segments
    )


# --------------------------------------------------------------------------- #
# end-to-end session plumbing (fake engine)
# --------------------------------------------------------------------------- #


def test_session_one_caption_per_frame_in_order():
    async def main():
        engine = FakeEngine()
        serving = _serving(engine)
        req = SessionRequest(system_prompt="output one integer")
        sess, frames, close = await _drive(serving, req, 3)
        assert [f.frame_index for f in frames] == [0, 1, 2]
        assert [f.text for f in frames] == ["frame0!", "frame1!", "frame2!"]
        assert all(f.finish_reason == "stop" for f in frames)
        assert all(f.token_count == 3 for f in frames)  # [1,2]+[3]
        assert all(f.ttft_s is not None and f.latency_s is not None for f in frames)
        assert close.frames == 3 and close.closed
        # first chunk carries the system prompt; later chunks are image-only
        assert "output one integer" in engine.seen_chunks[0].prompt["prompt"]
        assert "output one integer" not in engine.seen_chunks[1].prompt["prompt"]

    asyncio.run(main())


def test_validate_rejects_oversized_retention():
    async def main():
        # session requests max_video_segments=10 -> needs video budget >= 20 > 16
        serving = _serving(FakeEngine(video_limit=16))
        req = SessionRequest(
            system_prompt="S",
            retention=StreamingRetentionParams(max_video_segments=10),
        )
        with pytest.raises(StreamingError):
            await serving.create_session(req)

    asyncio.run(main())


def test_default_session_admitted_under_large_trained_range():
    # Regression: the re-trigger guard compares max_session_tokens against the
    # model's TRAINED position range (max_position_embeddings), NOT max_model_len.
    # An all-defaults SessionRequest must be admissible on a small-max_model_len
    # launch (8192) as long as the trained range is large (262144).
    async def main():
        engine = FakeEngine()
        engine.model_config = SimpleNamespace(
            model="fake-model",
            max_model_len=8192,
            hf_text_config=SimpleNamespace(max_position_embeddings=262144),
            multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda m: 64),
        )
        serving = _serving(engine)
        sess = await serving.create_session(SessionRequest(system_prompt="S"))
        assert sess.session_id

    asyncio.run(main())


def test_second_session_rejected_503_when_one_active():
    """Concurrent sessions are capped at 1: creating a second session while
    one is already open is rejected with 503, not silently queued."""

    async def main():
        serving = _serving(FakeEngine())
        first = await serving.create_session(SessionRequest(system_prompt="S"))
        assert first.session_id
        with pytest.raises(StreamingError) as ei:
            await serving.create_session(SessionRequest(system_prompt="S2"))
        assert ei.value.status_code == 503

    asyncio.run(main())


def test_conflicting_guided_flags_fail_fast_at_init():
    with pytest.raises(ValueError):
        _serving(
            FakeEngine(),
            sampling=SamplingConfig(guided_choice=["a"], guided_regex="x"),
        )


def test_push_unknown_session_404():
    async def main():
        serving = _serving(FakeEngine())
        with pytest.raises(StreamingError) as ei:
            await serving.push_frame("nope", _png_bytes())
        assert ei.value.status_code == 404

    asyncio.run(main())
