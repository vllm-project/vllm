# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for beam search with structured output.

Both the offline ``LLM`` path
(``vllm/entrypoints/generate/beam_search/offline.py``) and the online
OpenAI-compatible server
(``vllm/entrypoints/generate/beam_search/online.py``) need identical logic to
initialise a grammar backend and to compute per-beam allowed token IDs.
Keeping that logic here avoids duplication and makes future changes
easier.
"""

from __future__ import annotations

import json as _json
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.entrypoints.generate.beam_search.choice_trie import ChoiceTrie
from vllm.entrypoints.generate.beam_search.utils import BeamSearchSequence
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.bitmask import bitmask_to_token_ids
from vllm.v1.structured_output.request import get_structured_output_key


class BeamSearchSOState(NamedTuple):
    """Resolved structured-output state for a beam-search request.

    Exactly one path is populated: the CHOICE fast path sets ``trie`` and
    leaves ``backend``, ``key``, and ``bitmask`` as ``None``; the
    grammar/JSON/regex path sets ``backend``, ``key``, and ``bitmask`` and
    leaves ``trie`` as ``None``.
    """

    backend: StructuredOutputBackend | None
    key: tuple | None
    bitmask: Any
    trie: ChoiceTrie | None


def validate_and_resolve_beam_search_so(
    structured_outputs: StructuredOutputsParams,
    vllm_config: VllmConfig,
    tokenizer: TokenizerLike,
) -> None:
    """Validate a beam-search structured output request and resolve its backend.

    Beam search builds :class:`BeamSearchParams` directly and never runs the
    :class:`SamplingParams` validation that the non-beam path applies, so the
    same checks are replicated here. Validation rejects degenerate inputs
    (e.g. an empty choice list) and resolves ``structured_outputs._backend``
    from ``"auto"`` to a concrete backend using the identical fallback the
    non-beam path uses (xgrammar, falling back to guidance/outlines, with
    Tekken Mistral + Lark grammars routed to guidance).

    The resolution mutates ``structured_outputs`` in place.

    Args:
        structured_outputs: The structured output parameters from the request.
        vllm_config: The engine-level vLLM configuration.
        tokenizer: The tokenizer used by the model.

    Raises:
        VLLMValidationError: If the structured output request is invalid.

    """
    probe = SamplingParams(structured_outputs=structured_outputs)
    probe._validate_structured_outputs(
        model_config=vllm_config.model_config,
        structured_outputs_config=vllm_config.structured_outputs_config,
        tokenizer=tokenizer,
    )


def init_beam_search_so_backend(
    vllm_config: VllmConfig,
    tokenizer: TokenizerLike,
    vocab_size: int,
    structured_outputs: StructuredOutputsParams,
) -> BeamSearchSOState:
    """Initialise a structured output backend for beam search.

    For ``CHOICE``-type requests the xgrammar backend is bypassed entirely
    and a token-level prefix trie is built instead.  The trie is O(1) per
    step per beam and requires no FSM replay, so it is much faster for large
    choice sets.

    For all other request types the behaviour is identical to before: an
    xgrammar (or other configured) backend is instantiated and returned.

    Returns a :class:`BeamSearchSOState`.  Exactly one of the following is
    true:

    * ``trie is not None`` — CHOICE fast path; ``backend``, ``key``, and
      ``bitmask`` are all ``None``.
    * ``trie is None`` — grammar/JSON/regex path; ``backend``, ``key``, and
      ``bitmask`` are all set.

    Args:
        vllm_config: The engine-level vLLM configuration.
        tokenizer: The tokenizer used by the model.
        vocab_size: The model vocabulary size.
        structured_outputs: The structured output parameters from the
            request.

    Returns:
        A :class:`BeamSearchSOState`.

    Raises:
        ValueError: If the requested backend is not supported.

    """
    validate_and_resolve_beam_search_so(structured_outputs, vllm_config, tokenizer)

    key = get_structured_output_key(structured_outputs)
    request_type, grammar_spec = key

    # Fast path: build a prefix trie for CHOICE requests and skip xgrammar.
    if request_type == StructuredOutputOptions.CHOICE:
        choices = _json.loads(grammar_spec)
        eos_token_id = tokenizer.eos_token_id
        trie = ChoiceTrie.build(choices, tokenizer, eos_token_id=eos_token_id)
        return BeamSearchSOState(None, None, None, trie)

    # `_backend` is now a concrete backend resolved by the validator above.
    backend_name = structured_outputs._backend

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

    bitmask = backend.allocate_token_bitmask(1)

    return BeamSearchSOState(backend, key, bitmask, None)


def get_trie_allowed_token_ids(
    beam: BeamSearchSequence,
    trie: ChoiceTrie,
) -> list[int] | None:
    """Return the trie-allowed next token IDs for *beam*.

    Re-walks the trie from root over the beam's generated tokens each call.
    This is stateless — no per-beam trie pointer to maintain or clone.

    Returns ``None`` when the beam has completed a valid choice (terminal).
    Returns ``[]`` if the beam somehow went off-trie (defensive; shouldn't
    happen during constrained decoding).
    """
    orig = beam.orig_prompt
    if orig["type"] == "enc_dec":
        prompt_len = len(orig["decoder_prompt"]["prompt_token_ids"])
    else:
        prompt_len = len(orig["prompt_token_ids"])
    generated_ids = beam.tokens[prompt_len:]
    return trie.allowed_tokens_for(generated_ids)


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

    For ``CHOICE``-type requests, callers should use
    :func:`get_trie_allowed_token_ids` instead of this function to avoid
    the replay overhead.

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
