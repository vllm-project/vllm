# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla


class DeviceConfig(NamedTuple):
    available: bool
    backends: list[str]


# Maps each device to its availability and supported backends.
DEVICE_BACKENDS: dict[str, DeviceConfig] = {
    "cuda": DeviceConfig(
        available=current_platform.is_cuda()
        and current_platform.has_device_capability(80),
        # FlashInfer backend temporarily disabled due to invariant CTA sizes.
        # See FlashInfer issue #2424
        backends=["FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION"],
    ),
    "xpu": DeviceConfig(
        available=current_platform.is_xpu() and HAS_TRITON,
        backends=["TRITON_ATTN"],
    ),
    # ROCm reports device_type "cuda" but is_cuda() is False, so it needs its
    # own entry. The AITER and ROCm custom attention backends do not declare
    # supports_batch_invariance(), leaving the Triton backends.
    "rocm": DeviceConfig(
        available=current_platform.is_rocm() and HAS_TRITON,
        backends=["TRITON_ATTN"],
    ),
}

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
TEST_MODEL = os.getenv("VLLM_TEST_MODEL", DEFAULT_MODEL)

# Override backends for MLA models (MLA only supported on CUDA).
if os.getenv("VLLM_TEST_MODEL"):
    # Imported here, not at module scope. `model_arch_config_convertor` and
    # `vllm.config` import each other, so the convertor has to be reached
    # through `vllm.config`; importing it at the top of this file makes it the
    # first vllm import in an interpreter that imports a test module before
    # vllm itself, which dies with a partially-initialized-module ImportError.
    from vllm.transformers_utils.config import get_config
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    config = get_config(TEST_MODEL, trust_remote_code=False)
    if ModelArchConfigConvertorBase(config, config.get_text_config()).is_deepseek_mla():
        DEVICE_BACKENDS["cuda"] = DeviceConfig(
            available=DEVICE_BACKENDS["cuda"].available,
            backends=["TRITON_MLA"]
            + (["FLASH_ATTN_MLA"] if flash_attn_supports_mla() else []),
        )
        DEVICE_BACKENDS["xpu"] = DeviceConfig(
            available=DEVICE_BACKENDS["xpu"].available,
            backends=[],
        )
        DEVICE_BACKENDS["rocm"] = DeviceConfig(
            available=DEVICE_BACKENDS["rocm"].available,
            backends=["TRITON_MLA"],
        )

# Only include backends for devices that are actually available.
BACKENDS: list[str] = sorted(
    {b for cfg in DEVICE_BACKENDS.values() if cfg.available for b in cfg.backends}
)

skip_unsupported = pytest.mark.skipif(
    not any(cfg.available for cfg in DEVICE_BACKENDS.values()),
    reason="Requires CUDA >= Ampere (SM80), ROCm, or Intel XPU with Triton",
)

skip_if_not_cuda = pytest.mark.skipif(
    not DEVICE_BACKENDS["cuda"].available,
    reason="Requires CUDA >= Ampere (SM80)",
)

# For tests that only need a CUDA-alike GPU, i.e. anything whose kernels are
# Triton or HIP-portable rather than NVIDIA-specific.
skip_if_not_cuda_alike = pytest.mark.skipif(
    not (DEVICE_BACKENDS["cuda"].available or DEVICE_BACKENDS["rocm"].available),
    reason="Requires CUDA >= Ampere (SM80) or ROCm",
)


def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    # Generate more realistic prompts that will actually produce varied tokens
    # Use a mix of common English text patterns

    prompt_templates = [
        # Question-answer style
        "Question: What is the capital of France?\nAnswer: The capital of France is",
        "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
        "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
        # Story/narrative style
        "Once upon a time in a distant galaxy, there lived",
        "The old man walked slowly down the street, remembering",
        "In the year 2157, humanity finally discovered",
        # Technical/code style
        "To implement a binary search tree in Python, first we need to",
        "The algorithm works by iterating through the array and",
        "Here's how to optimize database queries using indexing:",
        # Factual/informative style
        "The Renaissance was a period in European history that",
        "Climate change is caused by several factors including",
        "The human brain contains approximately 86 billion neurons which",
        # Conversational style
        "I've been thinking about getting a new laptop because",
        "Yesterday I went to the store and bought",
        "My favorite thing about summer is definitely",
    ]

    # Pick a random template
    base_prompt = random.choice(prompt_templates)

    if max_words < min_words:
        max_words = min_words
    target_words = random.randint(min_words, max_words)

    if target_words > 50:
        # For longer prompts, repeat context
        padding_text = (
            " This is an interesting topic that deserves more explanation. "
            # TODO: Update to * (target_words // 10) to better align with word ratio
            * (target_words // 50)
        )
        base_prompt = padding_text + base_prompt

    return base_prompt


def _extract_step_logprobs(request_output):
    if getattr(request_output, "outputs", None):
        inner = request_output.outputs[0]
        if hasattr(inner, "logprobs") and inner.logprobs is not None:
            t = torch.tensor(
                [
                    inner.logprobs[i][tid].logprob
                    for i, tid in enumerate(inner.token_ids)
                ],
                dtype=torch.float32,
            )
            return t, inner.token_ids

    return None, None


def is_device_capability_below_90() -> bool:
    return not current_platform.has_device_capability(90)


def shutdown_llm(llm) -> None:
    """Tear an ``LLM`` down so the next model load has its VRAM back.

    ``LLM`` has no ``shutdown()`` method -- the engine-core child owns the
    memory and has to be asked directly. Deliberately not wrapped in
    ``contextlib.suppress``: a suppressed teardown that never ran is
    indistinguishable from one that worked.

    The VRAM comes back with the child a second or two after the caller drops
    its last reference to ``llm``, which is what the inter-module settle in
    ``conftest.py`` absorbs. Under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the
    engine still shuts down, but its VRAM is not recoverable in-process; run
    those tests in a spawned interpreter.
    """
    llm.llm_engine.engine_core.shutdown()


def assert_needle_is_batch_invariant(
    llm,
    *,
    padding_unit: str,
    padding_repeats: int,
    max_batch_size: int,
    max_tokens: int,
    num_trials: int,
    seed: int = 12345,
) -> None:
    """One fixed prompt's per-step logprobs must be equal at bs=1 and bs=N.

    The needle is never placed at batch index 0: that position keeps its token
    offset between the solo and the batched run, so it can stay invariant even
    when the rest of the batch does not. The fillers are staggered in length so
    they finish prefilling on different steps and the needle shares its forward
    passes with a changing mix of prefill and decode.
    """
    from vllm import SamplingParams

    assert max_batch_size >= 3, "Batch size should be >= 3 to place the needle."
    rng = random.Random(seed)
    sampling = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, seed=20240919, logprobs=1
    )
    needle_prompt = padding_unit * padding_repeats + (
        "Write one factual sentence about the moon."
    )

    baseline_output = llm.generate([needle_prompt], sampling, use_tqdm=False)[0]
    baseline_logprobs, baseline_token_ids = _extract_step_logprobs(baseline_output)
    assert baseline_logprobs is not None

    for _ in range(num_trials):
        batch_size = rng.randint(3, max_batch_size)
        needle_pos = rng.randint(1, batch_size - 1)
        prompts = []
        for idx in range(batch_size):
            if idx == needle_pos:
                prompts.append(needle_prompt)
                continue
            repeats = max(20, padding_repeats * (idx + 1) // batch_size)
            prompts.append(
                padding_unit * repeats + f"Describe topic number {idx} in detail."
            )

        needle_output = llm.generate(prompts, sampling, use_tqdm=False)[needle_pos]
        needle_logprobs, needle_token_ids = _extract_step_logprobs(needle_output)
        assert needle_logprobs is not None

        assert needle_output.prompt == needle_prompt
        assert needle_token_ids == baseline_token_ids
        assert torch.equal(needle_logprobs, baseline_logprobs), (
            f"Logprobs differ at needle position {needle_pos} of batch "
            f"{batch_size}: max |delta| = "
            f"{(needle_logprobs - baseline_logprobs).abs().max().item()}"
        )


def bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret ``t`` as integers, so comparisons are bitwise."""
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    return t.contiguous().view(view)


def rows_that_differ(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ne = bits(a) != bits(b)
    return torch.nonzero(ne.reshape(a.size(0), -1).any(dim=1)).flatten()


def order_sensitive_elements(probe: torch.Tensor) -> torch.Tensor:
    """Mask of probe elements whose reduction depends on the rank order.

    The collectives sum the ``world_size`` contributions of an element in rank
    order with an fp32 accumulator and round once on the way out, so an element
    can only notice a reordering if that accumulation is inexact for its
    operands. Summing the gathered contributions in the opposite order is the
    strongest reordering available and bounds what any other one can do: where
    it changes nothing, an invariance sweep cannot fail either.

    The all-gather is pure data movement, so every rank computes the same mask.
    """
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_tp_group

    world_size = get_tp_group().world_size
    gathered = torch.empty(
        (world_size * probe.shape[0], *probe.shape[1:]),
        dtype=probe.dtype,
        device=probe.device,
    )
    dist.all_gather_into_tensor(
        gathered, probe.contiguous(), group=get_tp_group().device_group
    )
    gathered = gathered.view(world_size, *probe.shape)

    ascending = torch.zeros(probe.shape, dtype=torch.float32, device=probe.device)
    for contribution in gathered:
        ascending += contribution.float()
    descending = torch.zeros_like(ascending)
    for contribution in gathered.flip(0):
        descending += contribution.float()
    return ascending.to(probe.dtype) != descending.to(probe.dtype)


INSTRUMENTATION_DIR = Path(__file__).parent / "instrumentation"


def instrumented_server_env(tmp_path, module: str, **extra: str) -> dict:
    """Env for a ``RemoteOpenAIServer`` that loads ``instrumentation/module``.

    A ``sitecustomize.py`` on the server's PYTHONPATH is the only hook that
    reaches the API server, every engine core and every worker; it shadows any
    other ``sitecustomize`` on the path. The instrumentation directory goes on
    the same PYTHONPATH so the shim can import from it.
    """
    (tmp_path / "sitecustomize.py").write_text(f"import {module}  # noqa: F401\n")
    return {
        "PYTHONPATH": os.pathsep.join(
            [
                str(tmp_path),
                str(INSTRUMENTATION_DIR),
                os.environ.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep),
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        **extra,
    }


def read_records(log_prefix: str) -> list[dict]:
    """Records written by the server-side instrumentation, one JSON per line.

    Each is tagged with the pid of the process that wrote it, since the ranks
    write to sibling files.
    """
    directory, prefix = os.path.split(log_prefix)
    out: list[dict] = []
    for name in os.listdir(directory):
        if not name.startswith(prefix + "."):
            continue
        pid = name.split(".", 1)[1]
        with open(os.path.join(directory, name)) as f:
            for line in f:
                if line.strip():
                    out.append({**json.loads(line), "pid": pid})
    return out


def assert_server_ran_this_mode(modes: set) -> None:
    """The server is a separate process, so it can run a different mode.

    An arm whose mode failed to propagate compares the server against itself and
    reports batch invariance without having varied anything.
    """
    assert modes == {envs.VLLM_BATCH_INVARIANT}, (
        f"the server's effective VLLM_BATCH_INVARIANT is {modes}, but this "
        f"process has {envs.VLLM_BATCH_INVARIANT}; the two arms of this test "
        "are not running the mode they claim to."
    )


def dp_completion(
    url: str,
    model: str,
    prompt,
    max_tokens: int,
    rank: int,
    *,
    logprobs: int | None = None,
    timeout: float = 900,
    extra_body: dict | None = None,
    extra_headers: dict | None = None,
) -> dict:
    """A greedy completion pinned to one data-parallel rank."""
    import requests

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 20240919,
        **(extra_body or {}),
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    headers = {"X-data-parallel-rank": str(rank), **(extra_headers or {})}
    response = requests.post(url, json=body, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


class BackgroundLoad:
    """Keeps ``concurrency`` requests in flight for the body of a ``with``.

    ``send(rng, index)`` issues one request; each worker thread gets its own
    seeded ``random.Random`` so the load is reproducible. Exceptions are
    collected in ``errors`` rather than raised, so a failing peer shows up as a
    single assertion in the test rather than a thread traceback. The ramp lets
    the server reach steady state before the caller measures, and the drain
    lets its queues empty so the next condition starts from idle.
    """

    def __init__(
        self,
        send,
        *,
        concurrency: int,
        ramp_seconds: float,
        drain_seconds: float,
        join_timeout: float = 300,
        seed: int = 0,
        prepare=None,
    ):
        self.send = send
        self.concurrency = concurrency
        self.ramp_seconds = ramp_seconds
        self.drain_seconds = drain_seconds
        self.join_timeout = join_timeout
        self.seed = seed
        self.prepare = prepare
        self.errors: list[str] = []
        self.completed = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def assert_ran_cleanly(self, label: str = "the background load") -> None:
        """No errors, and -- if companions were configured -- work actually done.

        A load that completes nothing raises nothing, so `errors` alone reads
        clean against an idle server.
        """
        assert not self.errors, f"{label} did not run cleanly: {self.errors[:3]}"
        assert self.completed or not self.concurrency, (
            f"{label} completed no requests, so the needle had no companions"
        )

    def _run(self, index: int) -> None:
        rng = random.Random(self.seed * 1000 + index)
        while not self._stop.is_set():
            try:
                self.send(rng, index)
                self.completed += 1
            except Exception as e:
                self.errors.append(repr(e))
                time.sleep(0.5)

    def __enter__(self) -> "BackgroundLoad":
        if self.concurrency and self.prepare is not None:
            self.prepare()
        for i in range(self.concurrency):
            thread = threading.Thread(target=self._run, args=(i,), daemon=True)
            thread.start()
            self._threads.append(thread)
        if self.concurrency:
            time.sleep(self.ramp_seconds)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=self.join_timeout)
        time.sleep(self.drain_seconds)
