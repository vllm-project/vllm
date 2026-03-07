# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for beam search with structured output.

Both the offline ``LLM`` path (``vllm/entrypoints/llm.py``) and the
online OpenAI-compatible server
(``vllm/entrypoints/openai/engine/serving.py``) need identical logic to
initialise a grammar backend and to compute per-beam allowed token IDs.
Keeping that logic here avoids duplication and makes future changes
easier.
"""

from __future__ import annotations

import json as _json
from typing import Any

from vllm.beam_search import BeamSearchSequence
from vllm.config import VllmConfig
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.bitmask import bitmask_to_token_ids
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.request import get_structured_output_key
from vllm.v1.structured_output.utils import choice_as_grammar


def init_beam_search_so_backend(
    vllm_config: VllmConfig,
    tokenizer: TokenizerLike,
    vocab_size: int,
    structured_outputs: StructuredOutputsParams,
) -> tuple[StructuredOutputBackend, tuple, Any]:
    """Initialise a structured output backend for beam search.

    Resolves the backend name from the engine's
    ``structured_outputs_config`` when the request does not specify one
    explicitly.  The returned tuple contains the instantiated backend,
    the grammar compilation key derived from the request parameters, and
    a pre-allocated single-row token bitmask.

    Args:
        vllm_config: The engine-level vLLM configuration.
        tokenizer: The tokenizer used by the model.
        vocab_size: The model vocabulary size.
        structured_outputs: The structured output parameters from the
            request.

    Returns:
        A ``(backend, key, bitmask)`` tuple ready for use in the beam
        search loop.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    so_config = vllm_config.structured_outputs_config

    # Resolve the backend name from engine config if not already set.
    if not structured_outputs._backend:
        structured_outputs._backend = so_config.backend

    backend_name = structured_outputs._backend

    # Resolve "auto" to a concrete backend.  The normal request path
    # does this during SamplingParams validation; beam search bypasses
    # that, so we replicate the resolution here.
    if backend_name == "auto":
        backend_name = "xgrammar"

    backend: StructuredOutputBackend
    if backend_name == "xgrammar":
        from vllm.v1.structured_output.backend_xgrammar import (
            XgrammarBackend,
        )

        backend = XgrammarBackend(
            vllm_config=vllm_config,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
    elif backend_name == "guidance":
        from vllm.v1.structured_output.backend_guidance import (
            GuidanceBackend,
        )

        backend = GuidanceBackend(
            vllm_config=vllm_config,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
    elif backend_name == "outlines":
        from vllm.v1.structured_output.backend_outlines import (
            OutlinesBackend,
        )

        backend = OutlinesBackend(
            vllm_config=vllm_config,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
    elif backend_name == "lm-format-enforcer":
        from vllm.v1.structured_output.backend_lm_format_enforcer import (
            LMFormatEnforcerBackend,
        )

        backend = LMFormatEnforcerBackend(
            vllm_config=vllm_config,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
        )
    else:
        raise ValueError(f"Unsupported structured output backend: {backend_name}")

    key = get_structured_output_key(structured_outputs)

    # Backends like xgrammar don't handle CHOICE natively in
    # compile_grammar — convert to an EBNF grammar first.
    request_type, grammar_spec = key
    if request_type == StructuredOutputOptions.CHOICE:
        choices = _json.loads(grammar_spec)
        key = (StructuredOutputOptions.GRAMMAR, choice_as_grammar(choices))

    bitmask = backend.allocate_token_bitmask(1)

    return backend, key, bitmask


def get_beam_allowed_token_ids(
    beam: BeamSearchSequence,
    backend: StructuredOutputBackend,
    so_key: tuple,
    bitmask: Any,
    vocab_size: int,
) -> list[int] | None:
    """Compute the set of grammar-allowed token IDs for a beam.

    A fresh grammar is compiled and all previously generated tokens are
    replayed through it so that the FSM state matches the beam's
    history.  This replay-per-step approach is necessary because
    structured output backends do not currently support cloning grammar
    state.

    Args:
        beam: The beam sequence whose grammar state to query.
        backend: The structured output backend instance.
        so_key: A ``(request_type, grammar_spec)`` tuple obtained from
            :func:`get_structured_output_key`.
        bitmask: A pre-allocated single-row bitmask tensor.
        vocab_size: The model vocabulary size.

    Returns:
        A list of allowed token IDs, or ``None`` when the grammar has
        terminated (the beam should be marked completed).
    """
    request_type, grammar_spec = so_key
    grammar = backend.compile_grammar(request_type, grammar_spec)

    orig = beam.orig_prompt
    if orig["type"] == "enc_dec":
        prompt_len = len(orig["decoder_prompt"]["prompt_token_ids"])
    else:
        prompt_len = len(orig["prompt_token_ids"])
    generated_tokens = beam.tokens[prompt_len:]

    if generated_tokens:
        grammar.accept_tokens("beam", generated_tokens)

    if grammar.is_terminated():
        return None

    grammar.fill_bitmask(bitmask, 0)
    allowed_ids = bitmask_to_token_ids(bitmask[0], vocab_size)

    return allowed_ids if allowed_ids else None
