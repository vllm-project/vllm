# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entrypoint tier of the KV-cache key-partitioning conformance suite.

``cache_salt`` partitions the prefix cache only if it survives the whole
trip from an HTTP request body down to the block hasher. That trip is what
keeps regressing: the salt has been dropped in chunked embeddings
preprocessing (#47696), across Harmony tool turns (#51818), and in external
cache keys (#51748). Each was found in the field, not by a test.

``tests/v1/core/test_prefix_caching.py`` asserts that differing salts
produce differing block hashes, but it constructs ``Request`` objects
directly, so it starts after the plumbing exercised here.

Every case asserts both directions. A test that only checks "different
salts do not share" also passes when prefix caching is disabled, and when
the salt reaches nothing and no request ever hits, so each negative arm is
paired with a positive control over the same prompt.

Assertions read the server's ``vllm:prefix_cache_hits`` counter rather than
``usage.prompt_tokens_details``, which is only populated when the server
runs with ``--enable-prompt-tokens-details``. Counters are also independent
of how the salt is encoded into the key, so this module is unaffected by
#44706, #51899, #52655 and #43789, each of which changes that encoding.

The counters are process-global, so these tests must run sequentially
against their own server. Do not run this module under pytest-xdist.

The pooling path gets its own server. Prefix caching applies to pooling
only for last-token or all pooling with causal attention, so the embeddings
arm runs a causal model under ``--runner pooling`` with ``LAST`` pooling
rather than a conventional bidirectional embedding model, which would never
engage the prefix cache at all. #47696 dropped the salt on that path.
"""

import os
import re

import pytest
import requests

from tests.utils import RemoteOpenAIServer

MODEL = "hmellor/tiny-random-LlamaForCausalLM"

# Long enough to span several blocks under any attention backend's block
# size. The block size is deliberately not pinned here: supported sizes
# differ per backend, so a fixed value fails on some of them, and nothing
# below needs to know it.
PROMPT_TOKENS = 256

SALT_A = "tenant-a-salt"
SALT_B = "tenant-b-salt"

# Long enough that the rendered chat prompt spans a full block whatever the
# template contributes. Chat cannot accept token ids, so this is the only
# prompt here whose length depends on the tokenizer.
CHAT_FILLER = "The quick brown fox jumps over the lazy dog. " * 12


@pytest.fixture(scope="module", autouse=True)
def _require_serial_execution():
    """The counters are process-global, so parallel workers corrupt deltas.

    The docstring above asks for serial execution and CI happens to honour it,
    but nothing enforced it. Fail loudly rather than silently mis-measuring.
    """
    assert "PYTEST_XDIST_WORKER" not in os.environ, (
        "This module reads process-global prefix-cache counters and must run "
        "serially. Run it without -n / xdist."
    )


@pytest.fixture(scope="module")
def server():
    args = [
        "--enforce-eager",
        "--enable-prefix-caching",
        "--max-model-len",
        "1024",
        "--load-format",
        "dummy",
        # On the CPU backend this flag sizes the KV cache out of host
        # RAM rather than device memory. The model here is tiny, so a
        # small fraction leaves room for whatever else is running.
        "--gpu-memory-utilization",
        "0.15",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


def _read_hits(server: RemoteOpenAIServer) -> int:
    """Tokens served from the prefix cache since the server started.

    ``prometheus_client`` appends ``_total`` to counter names on exposition,
    and the counter carries per-engine labels, so accept either spelling and
    sum across label sets rather than pinning a form this module does not
    control.
    """
    response = requests.get(server.url_for("metrics"), timeout=30)
    response.raise_for_status()
    pattern = re.compile(
        r"^vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
        re.MULTILINE,
    )
    matches = pattern.findall(response.text)
    assert matches, "vllm:prefix_cache_hits is not exposed on /metrics"
    return int(sum(float(value) for value in matches))


def _post(server: RemoteOpenAIServer, path: str, payload: dict) -> None:
    response = requests.post(server.url_for(path), json=payload, timeout=60)
    response.raise_for_status()


def _prompt_ids(marker: int) -> list[int]:
    """A prompt no other case in this module shares.

    All cases share one server, so a prompt reused across cases would
    already be warm and a negative arm could report a hit it did not cause.
    """
    return [marker] * PROMPT_TOKENS


def _completions_body(prompt_ids: list[int], salt: str | None) -> dict:
    body: dict = {
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
    }
    if salt is not None:
        body["cache_salt"] = salt
    return body


def _chat_body(content: str, salt: str) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0.0,
        "cache_salt": salt,
    }


def _assert_no_reuse(delta: int, case: str) -> None:
    assert delta == 0, (
        f"{case}: {delta} tokens were served from the prefix cache to a "
        f"request whose cache_salt differed from the one that warmed it. "
        f"Blocks are crossing the salt boundary."
    )


def _assert_reuse(delta: int, case: str) -> None:
    assert delta > 0, (
        f"{case}: an identical request under the same cache_salt reused no "
        f"cached tokens at all. Without this control the isolation assertion "
        f"above is vacuous, since it would also pass with prefix caching "
        f"switched off, or with a salt that reaches nothing."
    )


def test_completions_isolates_on_salt(server):
    """Token-id prompts, so the block count does not depend on a tokenizer."""
    prompt_ids = _prompt_ids(101)

    _post(server, "v1/completions", _completions_body(prompt_ids, SALT_A))

    before = _read_hits(server)
    _post(server, "v1/completions", _completions_body(prompt_ids, SALT_B))
    _assert_no_reuse(_read_hits(server) - before, "completions")

    before = _read_hits(server)
    _post(server, "v1/completions", _completions_body(prompt_ids, SALT_A))
    _assert_reuse(_read_hits(server) - before, "completions")


def test_chat_completions_isolates_on_salt(server):
    """The chat path renders a template before tokenizing; the salt must
    still partition whatever that rendering produces."""
    _post(server, "v1/chat/completions", _chat_body(CHAT_FILLER, SALT_A))

    before = _read_hits(server)
    _post(server, "v1/chat/completions", _chat_body(CHAT_FILLER, SALT_B))
    _assert_no_reuse(_read_hits(server) - before, "chat/completions")

    before = _read_hits(server)
    _post(server, "v1/chat/completions", _chat_body(CHAT_FILLER, SALT_A))
    _assert_reuse(_read_hits(server) - before, "chat/completions")


def test_absent_salt_does_not_match_salted_prefix(server):
    """A request that omits cache_salt must not land in a salted namespace.

    Omission is the default for every client that has not been updated, so a
    deployment that salts some callers and not others must not silently share
    between them.
    """
    prompt_ids = _prompt_ids(102)

    _post(server, "v1/completions", _completions_body(prompt_ids, SALT_A))

    before = _read_hits(server)
    _post(server, "v1/completions", _completions_body(prompt_ids, None))
    _assert_no_reuse(_read_hits(server) - before, "completions, salt omitted")

    # Positive control for the arm above: two unsalted requests share with
    # each other, so the miss asserted there is the salt boundary holding
    # rather than the unsalted path failing to cache at all.
    before = _read_hits(server)
    _post(server, "v1/completions", _completions_body(prompt_ids, None))
    _assert_reuse(_read_hits(server) - before, "completions, both unsalted")


@pytest.fixture(scope="module")
def pooling_server():
    """A pooling server that can actually reach the prefix cache.

    Prefix caching applies to pooling only for last-token or all pooling
    with causal attention, so this serves a causal model with ``LAST``
    pooling. A conventional bidirectional embedding model would never
    engage the cache, and the arm below would pass without testing
    anything.
    """
    args = [
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--pooler-config",
        '{"pooling_type":"LAST"}',
        "--enforce-eager",
        "--enable-prefix-caching",
        "--max-model-len",
        "1024",
        "--load-format",
        "dummy",
        "--gpu-memory-utilization",
        "0.15",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


def _embeddings_body(prompt_ids: list[int], salt: str | None) -> dict:
    body: dict = {"model": MODEL, "input": prompt_ids}
    if salt is not None:
        body["cache_salt"] = salt
    return body


def test_embeddings_isolates_on_salt(pooling_server):
    """The pooling path rebuilds prompts before they reach the hasher.

    #47696 dropped the salt in exactly that rebuild, so a second
    ``/v1/embeddings`` request under a different salt could still reuse the
    first request's cached chunks.
    """
    prompt_ids = _prompt_ids(201)

    _post(pooling_server, "v1/embeddings", _embeddings_body(prompt_ids, SALT_A))

    before = _read_hits(pooling_server)
    _post(pooling_server, "v1/embeddings", _embeddings_body(prompt_ids, SALT_B))
    _assert_no_reuse(_read_hits(pooling_server) - before, "embeddings")

    before = _read_hits(pooling_server)
    _post(pooling_server, "v1/embeddings", _embeddings_body(prompt_ids, SALT_A))
    _assert_reuse(_read_hits(pooling_server) - before, "embeddings")


@pytest.fixture(scope="module")
def chunked_pooling_server():
    """A pooling server with chunked processing switched on.

    ``enable_chunked_processing`` routes every pooling request through
    ``maybe_pre_process_chunked``, which rebuilds the prompt before it
    reaches the hasher. That rebuild is where #47696 drops the salt, so the
    default-off server above never exercises the path the bug lives on.
    """
    args = [
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--pooler-config",
        '{"pooling_type":"LAST","enable_chunked_processing":true}',
        "--enforce-eager",
        "--enable-prefix-caching",
        "--max-model-len",
        "1024",
        "--load-format",
        "dummy",
        "--gpu-memory-utilization",
        "0.15",
    ]
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


@pytest.mark.xfail(
    strict=True,
    reason="#47696: chunked pooling preprocessing drops request-level cache_salt",
)
def test_chunked_embeddings_isolate_on_salt(chunked_pooling_server):
    """Isolation must survive the chunked-preprocessing rebuild.

    Note the prompt here is shorter than ``max_model_len``, so it is never
    actually split. Enabling the flag is sufficient: every request takes the
    rebuild path, and the salt is lost there whether or not chunking occurs.
    That is wider than #47696's description, which frames the loss as
    happening to long inputs.
    """
    server = chunked_pooling_server
    prompt_ids = _prompt_ids(202)

    _post(server, "v1/embeddings", _embeddings_body(prompt_ids, SALT_A))

    before = _read_hits(server)
    _post(server, "v1/embeddings", _embeddings_body(prompt_ids, SALT_B))
    _assert_no_reuse(_read_hits(server) - before, "embeddings, chunked processing")

    before = _read_hits(server)
    _post(server, "v1/embeddings", _embeddings_body(prompt_ids, SALT_A))
    _assert_reuse(_read_hits(server) - before, "embeddings, chunked processing")


@pytest.mark.xfail(
    strict=True,
    reason="#56199: pooling accepts an empty cache_salt and shares the "
    "unsalted namespace",
)
def test_empty_salt_does_not_match_unsalted_prefix(pooling_server):
    """An empty ``cache_salt`` must not be treated as no salt at all.

    Every other protocol accepting ``cache_salt`` pins ``min_length=1``; the
    pooling mixin does not, and ``generate_block_hash_extra_keys`` tests the
    salt for truthiness rather than for ``None``. So ``""`` produces no extra
    key and lands in the unsalted namespace, which a client emitting an empty
    string by mistake would do silently.

    Rejecting it with a 400, as #56199 does, is the behaviour this pins.
    """
    server = pooling_server
    prompt_ids = _prompt_ids(203)

    _post(server, "v1/embeddings", _embeddings_body(prompt_ids, None))

    before = _read_hits(server)
    response = requests.post(
        server.url_for("v1/embeddings"),
        json=_embeddings_body(prompt_ids, ""),
        timeout=60,
    )
    # 400 rather than FastAPI's default 422: vLLM registers a
    # RequestValidationError handler mapping body validation failures to
    # BAD_REQUEST with the OpenAI error envelope, which existing pooling
    # tests assert too.
    assert response.status_code == 400, (
        f"an empty cache_salt was accepted with {response.status_code}; it "
        f"should be refused like it is on every other endpoint"
    )
    _assert_no_reuse(_read_hits(server) - before, "embeddings, empty salt")
