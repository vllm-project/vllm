# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional end-of-sequence hook for `vllm.endpoint_plugins`.

Classifier / guardrail plugins implement ``async def post_generation`` on an
``EndpointPlugin``. The serving layer invokes it once after the final
``RequestOutput`` and before OpenAI serialization (RFC #43999). Plugins that
only attach HTTP routes omit the method and are skipped.

This runs in the API server process only. Endpoint plugins are already gated
by ``VLLM_PLUGINS`` and are not loaded in engine-core or worker processes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 1000


@dataclass
class ScoringContext:
    """Detokenized generation plus request extras for a post-generation hook."""

    request_id: str
    prompt: str | None
    generated_text: str
    finish_reason: str | None
    prompt_token_ids: list[int] | None
    output_token_ids: list[int]
    vllm_xargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostGenerationOutcome:
    """Result of running all loaded post-generation hooks."""

    blocked: bool = False
    replacement: str | None = None
    error_message: str | None = None
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)


def scoring_context_from_request_output(
    request_output: RequestOutput,
    vllm_xargs: Mapping[str, Any] | None = None,
) -> ScoringContext:
    outputs = request_output.outputs
    generated_text = "\n".join(output.text for output in outputs)
    finish_reason = outputs[0].finish_reason if outputs else None
    output_token_ids: list[int] = []
    for output in outputs:
        output_token_ids.extend(int(tid) for tid in output.token_ids)
    return ScoringContext(
        request_id=request_output.request_id,
        prompt=request_output.prompt,
        generated_text=generated_text,
        finish_reason=finish_reason,
        prompt_token_ids=request_output.prompt_token_ids,
        output_token_ids=output_token_ids,
        vllm_xargs=dict(vllm_xargs or {}),
    )


def _apply_replacement(request_output: RequestOutput, replacement: str) -> None:
    for output in request_output.outputs:
        output.text = replacement


async def apply_post_generation_hooks(
    plugins: Sequence[Any],
    request_output: RequestOutput,
    vllm_xargs: Mapping[str, Any] | None = None,
) -> PostGenerationOutcome:
    """Run optional ``post_generation`` methods on loaded endpoint plugins.

    Annotation scores are stored on ``request_output.metadata["external_scores"]``.
    A blocking plugin may replace generated text or refuse the request.
    """
    outcome = PostGenerationOutcome()
    if not plugins:
        return outcome

    ctx = scoring_context_from_request_output(request_output, vllm_xargs)
    scores: dict[str, dict[str, Any]] = {}

    for plugin in plugins:
        hook = getattr(plugin, "post_generation", None)
        if hook is None:
            continue

        name = getattr(plugin, "name", plugin.__class__.__name__)
        blocking = bool(getattr(plugin, "blocking", False))
        timeout_ms = getattr(plugin, "timeout_ms", DEFAULT_TIMEOUT_MS)
        try:
            timeout_s = max(float(timeout_ms), 0.0) / 1000.0
        except (TypeError, ValueError):
            timeout_s = DEFAULT_TIMEOUT_MS / 1000.0

        try:
            raw = await asyncio.wait_for(hook(ctx), timeout=timeout_s)
        except TimeoutError:
            logger.warning(
                "Post-generation hook %r timed out after %.0f ms",
                name,
                timeout_s * 1000,
            )
            if blocking:
                outcome.blocked = True
                outcome.error_message = f"post-generation hook {name!r} timed out"
                break
            continue
        except Exception:
            logger.exception("Post-generation hook %r failed", name)
            if blocking:
                outcome.blocked = True
                outcome.error_message = f"post-generation hook {name!r} failed"
                break
            continue

        if not isinstance(raw, dict):
            logger.warning(
                "Post-generation hook %r returned %s, expected dict",
                name,
                type(raw).__name__,
            )
            continue

        scores[name] = raw
        if blocking and raw.get("block"):
            outcome.blocked = True
            replacement = raw.get("replacement")
            if isinstance(replacement, str):
                outcome.replacement = replacement
                _apply_replacement(request_output, replacement)
            else:
                message = raw.get("message")
                outcome.error_message = (
                    message
                    if isinstance(message, str) and message
                    else f"blocked by post-generation hook {name!r}"
                )
            break

    if scores:
        metadata = request_output.metadata
        if metadata is None:
            metadata = {}
            request_output.metadata = metadata
        external = metadata.setdefault("external_scores", {})
        external.update(scores)
        outcome.scores = dict(external)

    return outcome
