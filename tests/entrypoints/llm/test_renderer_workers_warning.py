# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for vllm-project/vllm#42901.

The renderer thread pool sized by `renderer_num_workers` is only consumed
by the async renderer path (e.g. `vllm serve` / `AsyncLLM`). The offline
`LLM` entrypoint uses the synchronous renderer path and runs multimodal
preprocessing serially, so `renderer_num_workers > 1` is a silent no-op
there. `LLM.__init__` should warn in that case.
"""

import logging

from vllm import LLM


def test_renderer_num_workers_warns_on_offline_llm(caplog):
    with caplog.at_level(logging.WARNING, logger="vllm.entrypoints.llm"):
        LLM(
            model="openai-community/gpt2",
            enforce_eager=True,
            renderer_num_workers=2,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "renderer_num_workers" in msg and "offline" in msg.lower()
        for msg in messages
    ), (
        "Expected a warning about renderer_num_workers having no effect "
        f"on the offline LLM path; saw: {messages}"
    )


def test_renderer_num_workers_default_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="vllm.entrypoints.llm"):
        LLM(model="openai-community/gpt2", enforce_eager=True)

    messages = [record.getMessage() for record in caplog.records]
    assert not any("renderer_num_workers" in msg for msg in messages), (
        "Did not expect a renderer_num_workers warning at the default "
        f"value of 1; saw: {messages}"
    )
