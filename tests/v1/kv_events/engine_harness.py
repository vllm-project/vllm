# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared pieces of the KV-event end-to-end tests.

A :class:`ServerConfig` describes one ``vllm serve`` (model, cache layout,
speculative decoding); :func:`serve` starts it with KV events enabled and
returns a :class:`ServedEngine` that sends requests and compares the engine's
``cached_tokens`` with the reference consumer's prediction.
"""

import json
import math
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest
import regex as re
import requests
import torch

from tests.utils import RemoteOpenAIServer

from .reference_consumer import EventStream

REQUEST_TIMEOUT_S = 300
GIB = 1024**3
# When set, ``<dir>/<org>/<name>`` is used instead of downloading a model,
# for hosts that keep weights on shared storage.
MODEL_DIR_ENV = "VLLM_KV_EVENTS_MODEL_DIR"


@dataclass(frozen=True)
class ServerConfig:
    name: str
    model: str
    # ``--block-size``. Hybrid models end up with much larger cache blocks:
    # the attention block grows until its page matches the Mamba state page.
    # Tests therefore size prompts in the block size learned from the events
    # (``ServedEngine.block_size``), not in this one.
    block_size: int = 16
    # Small KV caches make eviction and preemption happen with short prompts.
    num_gpu_blocks: int = 96
    # Prompt length in (learned) blocks for the single-request checks.
    prompt_blocks: int = 10
    # Prompt length in blocks for the concurrent-requests check.
    concurrent_blocks: int = 12
    # Whether the concurrent-requests check can rely on preemption happening.
    expect_preemption: bool = True
    # Free GPU memory the server needs; the test is skipped below that.
    gpu_gib: float = 8.0
    gpus: int = 1
    max_model_len: int = 1024
    max_num_seqs: int = 8
    # ``--speculative-config`` contents; a draft ``model`` is resolved like
    # the target model.
    speculative: dict | None = None
    extra_args: tuple[str, ...] = ()
    start_timeout_s: float = 600
    description: str = ""

    @property
    def eagle_drop(self) -> bool:
        """EAGLE-style drafters read one token past the cached prefix: the
        engine hashes that token into each block or, with lookahead hashes
        disabled, drops the last matched block. The reference consumer tells
        the two apart from the events."""
        method = (self.speculative or {}).get("method")
        return method in ("eagle", "eagle3", "mtp")


def resolve_model(name: str) -> str:
    root = os.environ.get(MODEL_DIR_ENV)
    if root:
        local = os.path.join(root, name)
        if os.path.isdir(local):
            return local
    return name


def gpu_memory_utilization(config: ServerConfig) -> str:
    free, total = torch.accelerator.get_memory_info()
    needed = config.gpu_gib * GIB
    if free - GIB < needed:
        pytest.skip(
            f"{config.name} needs {config.gpu_gib:.0f} GiB of free GPU memory, "
            f"{free / GIB:.1f} GiB available"
        )
    return f"{max(0.03, needed / total):.3f}"


def server_args(
    config: ServerConfig, pub_port: int, replay_port: int, buffer_steps: int
) -> list[str]:
    kv_events = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        # The publisher only binds wildcard endpoints; a host:port makes it
        # connect instead.
        "endpoint": f"tcp://*:{pub_port}",
        "replay_endpoint": f"tcp://127.0.0.1:{replay_port}",
        "buffer_steps": buffer_steps,
        "topic": "",
    }
    args = [
        "--enforce-eager",
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--block-size",
        str(config.block_size),
        "--num-gpu-blocks-override",
        str(config.num_gpu_blocks),
        "--gpu-memory-utilization",
        gpu_memory_utilization(config),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--kv-events-config",
        json.dumps(kv_events),
    ]
    if config.speculative is not None:
        speculative = dict(config.speculative)
        if "model" in speculative:
            speculative["model"] = resolve_model(speculative["model"])
        args += ["--speculative-config", json.dumps(speculative)]
    args += list(config.extra_args)
    return args


def random_ids(seed: int, length: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 100_000) for _ in range(length)]


def prompt_ids(
    seed: int, num_blocks: int, block_size: int, extra_tokens: int = 3
) -> list[int]:
    """Deterministic token ids; ``extra_tokens`` leaves a partial last block."""
    return random_ids(seed, num_blocks * block_size + extra_tokens)


def repetitive_prompt_ids(
    seed: int, num_blocks: int, block_size: int, period: int = 40
) -> list[int]:
    """A repeated random pattern: greedy decoding continues it, which gives
    n-gram speculation something to propose."""
    pattern = random_ids(seed, period)
    length = num_blocks * block_size + 3
    return (pattern * (length // period + 1))[:length]


@contextmanager
def subscribe(pub_port: int, replay_port: int) -> Iterator[EventStream]:
    stream = EventStream(
        f"tcp://127.0.0.1:{pub_port}", f"tcp://127.0.0.1:{replay_port}"
    )
    try:
        yield stream
    finally:
        stream.close()


@dataclass
class ServedEngine:
    config: ServerConfig
    server: RemoteOpenAIServer
    model_name: str
    pub_port: int
    replay_port: int
    # Learned from the events of a calibration request (see ``calibrate``).
    block_size: int = 0
    groups: str = field(default="", repr=False)

    def prompt(self, seed: int, num_blocks: int, extra_tokens: int = 3) -> list[int]:
        return prompt_ids(seed, num_blocks, self.block_size, extra_tokens)

    def complete(self, token_ids: list[int], max_tokens: int = 1, **extra) -> dict:
        body = {
            "model": self.model_name,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            **extra,
        }
        response = requests.post(
            self.server.url_for("v1/completions"), json=body, timeout=REQUEST_TIMEOUT_S
        )
        response.raise_for_status()
        return response.json()

    def reset_prefix_cache(self) -> bool:
        response = requests.post(self.server.url_for("reset_prefix_cache"), timeout=30)
        response.raise_for_status()
        return response.json()["success"]

    def metric(self, name: str) -> float:
        """Sum of a counter over all its label sets in ``/metrics``."""
        text = requests.get(self.server.url_for("metrics"), timeout=30).text
        pattern = rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)"
        return sum(float(m.group(1)) for m in re.finditer(pattern, text, re.M))

    def num_preemptions(self) -> float:
        return self.metric("vllm:num_preemptions_total")

    @contextmanager
    def subscribe(self) -> Iterator[EventStream]:
        with subscribe(self.pub_port, self.replay_port) as stream:
            yield stream

    def calibrate(self) -> None:
        """Learn the engine's real cache block size from the events of one
        long request, then clear the cache again."""
        with self.subscribe() as stream:
            self.complete(random_ids(0, self.config.max_model_len - 1))
            stream.drain()
            groups = stream.index.groups
            assert groups, (
                f"{self.config.name}: a {self.config.max_model_len - 1}-token "
                "request published no BlockStored event"
            )
            self.block_size = math.lcm(*(group.block_size for group in groups.values()))
            self.groups = stream.index.describe_groups()
            assert self.reset_prefix_cache()
            # The reset is published with the engine's next step.
            self.complete(random_ids(1, 8))
            stream.drain()

    def check(self, stream: EventStream, token_ids: list[int], label: str) -> int:
        """Predict the hit from the events seen so far, send the request,
        compare. Returns the ``cached_tokens`` the engine reported."""
        expected = stream.index.predict_cached_tokens(
            token_ids, eagle_drop=self.config.eagle_drop
        )
        actual = cached_tokens(self.complete(token_ids))
        stream.drain()
        assert actual == expected, (
            f"{self.config.name}, {label}: engine reported cached_tokens="
            f"{actual}, the event-derived index predicted {expected} "
            f"({stream.index.describe_groups()})"
        )
        return actual

    def fresh_state(self, stream: EventStream) -> None:
        """Reset the prefix cache and make sure the consumer saw the reset.

        A throwaway request forces publication of whatever the engine has
        queued, so this does not depend on idle resets being published.
        """
        assert self.reset_prefix_cache()
        self.complete(random_ids(1, 8))
        stream.drain()
        assert stream.index.stats["cleared"] >= 1
        stream.index.anomalies.clear()


def cached_tokens(completion: dict) -> int:
    return completion["usage"]["prompt_tokens_details"]["cached_tokens"]


@contextmanager
def serve(
    config: ServerConfig, pub_port: int, replay_port: int, buffer_steps: int = 10_000
) -> Iterator[ServedEngine]:
    if torch.accelerator.device_count() < config.gpus:
        pytest.skip(f"{config.name} needs {config.gpus} GPUs")
    model = resolve_model(config.model)
    # Dev mode exposes /reset_prefix_cache.
    with RemoteOpenAIServer(
        model,
        server_args(config, pub_port, replay_port, buffer_steps),
        env_dict={"VLLM_SERVER_DEV_MODE": "1"},
        max_wait_seconds=config.start_timeout_s,
    ) as server:
        engine = ServedEngine(config, server, model, pub_port, replay_port)
        engine.calibrate()
        yield engine
