# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact batch shapes for `vllm bench forward`."""

import json
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import regex as re


@dataclass(frozen=True)
class BenchRequest:
    query_len: int
    # KV length after the step: the prefix already in the KV cache plus the
    # query tokens.
    kv_len: int
    # Spec-decode verify: query_len == 1 + num_draft_tokens.
    num_draft_tokens: int = 0
    # The prefix was computed by this request's earlier chunks. Otherwise a
    # prefill's prefix is a prefix-cache hit, of which models with SWA bounded
    # replay (DeepSeek-V4.1) recompute the last window.
    chunked: bool = False

    def __post_init__(self) -> None:
        if (
            self.query_len < 1
            or self.kv_len < self.query_len
            or self.num_draft_tokens < 0
        ):
            raise ValueError(f"Invalid request: {self}")
        if self.num_draft_tokens and not self.is_decode:
            raise ValueError(
                f"Verify needs query_len == 1 + num_draft_tokens and a prefix: {self}"
            )
        if self.chunked and (self.is_decode or not self.prefix_len):
            raise ValueError(f"A chunk needs a prefill with a prefix: {self}")

    @property
    def prefix_len(self) -> int:
        return self.kv_len - self.query_len

    @property
    def is_decode(self) -> bool:
        """Decode or verify; otherwise a prefill."""
        return self.prefix_len > 0 and self.query_len == 1 + self.num_draft_tokens

    @property
    def prefix_hit(self) -> bool:
        return self.prefix_len > 0 and not self.is_decode and not self.chunked


@dataclass(frozen=True)
class BenchCase:
    name: str
    requests: tuple[BenchRequest, ...]

    @property
    def num_tokens(self) -> int:
        return sum(r.query_len for r in self.requests)


def jitter_kv_lens(
    case: BenchCase, jitter: float, max_kv_len: int, seed: int
) -> BenchCase:
    """Draw each request's prefix uniformly within +-jitter of its value.

    Query lengths are kept, so a fresh prefill stays one and a non-empty
    prefix stays non-empty. KV lengths are capped at `max_kv_len`. The draw
    depends only on the seed and the case's name, not on the other cases.
    """
    rng = np.random.default_rng([seed, zlib.crc32(case.name.encode())])
    requests = []
    for req in case.requests:
        if req.kv_len > max_kv_len:
            raise ValueError(f"{case.name}: KV length {req.kv_len} > {max_kv_len}")
        prefix = req.prefix_len
        if prefix and jitter:
            prefix = max(1, round(prefix * rng.uniform(1 - jitter, 1 + jitter)))
        kv_len = min(req.query_len + prefix, max_kv_len)
        requests.append(replace(req, kv_len=kv_len))
    return replace(case, requests=tuple(requests))


_REQ_RE = re.compile(r"^(?:(\d+)\*)?(\d+)(?:@(\d+))?(c)?(?::(\d+))?$")


def parse_case(text: str) -> BenchCase:
    """Parse `NAME=REQS`, where REQS is a comma list of `[N*]Q[@K][c][:D]`.

    Q is the query length, K the KV length including the query (default Q, a
    fresh prefill), D the number of draft tokens (verify) and N the number of
    such requests. A prefill's prefix is a prefix-cache hit unless `c` marks
    it as a chunked prefill. For example, `decode=32*1@8192`,
    `verify=16*4@4096:3`, `prefill=8192`, `hit=8192@40960`,
    `chunk=8192@40960c` or `mix=8192,32*1@8192`.
    """
    name, sep, body = text.partition("=")
    if not sep:
        name, body = text, text
    requests: list[BenchRequest] = []
    for item in body.split(","):
        match = _REQ_RE.match(item.strip())
        if match is None:
            raise ValueError(f"Invalid request {item!r} in case {text!r}")
        count, query, kv, chunked, drafts = match.groups()
        request = BenchRequest(
            int(query), int(kv or query), int(drafts or 0), chunked=bool(chunked)
        )
        requests.extend([request] * int(count or 1))
    return BenchCase(name, tuple(requests))


def load_cases(path: str) -> list[BenchCase]:
    """Load cases from a JSON or YAML file.

    Format: `{"cases": [{"name": ..., "requests": [{"query_len": ...,
    "kv_len": ..., "num_draft_tokens": ..., "chunked": ..., "count": ...}]}]}`.
    """
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    cases = []
    for case in data["cases"]:
        requests: list[BenchRequest] = []
        for req in case["requests"]:
            req = dict(req)
            count = req.pop("count", 1)
            requests.extend([BenchRequest(**req)] * count)
        cases.append(BenchCase(case["name"], tuple(requests)))
    return cases
