# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""REST streaming API.

Exposes the per-frame streaming-video caption loop (one rendered frame in, one
text report out, with engine-side rolling KV/encoder/mRoPE retention) over a
small stateful HTTP API so external harnesses can drive the real model:

    POST   /v1/streaming/sessions
        {system_prompt, retention, sampling, fps} -> {session_id}
    POST   /v1/streaming/sessions/{id}/frame
        <image bytes> -> {frame_index, text, ...}
    DELETE /v1/streaming/sessions/{id}
        -> {closed}

Each session is one long-lived ``engine.generate(prompt=<async-gen of
StreamingInput>, request_id=...)`` call; the rolling state lives engine-side,
tied to that request, not to any HTTP request. See ``examples/streaming/`` for
a runnable client.
"""
