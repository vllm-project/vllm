# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 multi-turn prefix cache reuse with KV offload, P/D and DCP."""

import contextlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from math import lcm

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.models.utils import check_logprobs_close
from tests.utils import RemoteOpenAIServer, multi_gpu_marks
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

# Official Kimi-K3 config with 8 layers and no weights.
MODEL = "riverclouds/Kimi-K3-8L-dummy"

# Conversation: the first prompt spans FIRST_PROMPT_BLOCKS blocks plus an
# unaligned tail, and each later turn appends TURN_TOKENS random tokens.
NUM_TURNS = 3
FIRST_PROMPT_BLOCKS = 3
UNALIGNED_TAIL = 37
TURN_TOKENS = 777
MAX_TOKENS = 8
NUM_LOGPROBS = 5
STARTUP_TIMEOUT = 1200

BASE_ARGS = [
    "--trust-remote-code",
    "--language-model-only",
    "--load-format",
    "dummy",
    "--no-enable-flashinfer-autotune",
    "--enable-prompt-tokens-details",
    # Default max_num_seqs OOMs one GPU during warmup.
    "--max-num-seqs",
    "8",
]

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Kimi-K3 NVIDIA kernels require the SM100 family",
)

NIXL = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
OFFLOAD = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 32 << 30},
}
SIMPLE_OFFLOAD = {
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 32 << 30},
}
NIXL_OFFLOAD = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"connectors": [NIXL, OFFLOAD]},
}


@dataclass(frozen=True)
class Instance:
    tp: int = 1
    dcp: int = 1
    kv_config: dict | None = None
    prefix_caching: bool = True

    def args(self, prefix_match_unit: int | None) -> list[str]:
        args = BASE_ARGS + ["--tensor-parallel-size", str(self.tp)]
        if self.dcp > 1:
            args += ["--decode-context-parallel-size", str(self.dcp)]
        if self.kv_config is not None:
            args += ["--kv-transfer-config", json.dumps(self.kv_config)]
        if not self.prefix_caching:
            return args + ["--no-enable-prefix-caching"]
        args.append("--enable-prefix-caching")
        if prefix_match_unit is not None:
            args += ["--prefix-match-unit", str(prefix_match_unit)]
        return args


@dataclass(frozen=True)
class Deployment:
    decode: Instance
    prefill: Instance | None = None
    # Reset GPU prefix caches each turn so hits must come from CPU.
    offload: bool = False

    @property
    def instances(self) -> tuple[Instance, ...]:
        return (self.prefill, self.decode) if self.prefill else (self.decode,)

    @property
    def num_gpus(self) -> int:
        return sum(i.tp for i in self.instances)


DEPLOYMENTS = {
    "plain": Deployment(Instance()),
    "offload": Deployment(Instance(kv_config=OFFLOAD), offload=True),
    "simple-offload": Deployment(Instance(kv_config=SIMPLE_OFFLOAD), offload=True),
    "dcp2": Deployment(Instance(tp=2, dcp=2)),
    "dcp2-offload": Deployment(Instance(tp=2, dcp=2, kv_config=OFFLOAD), offload=True),
    "pd": Deployment(Instance(kv_config=NIXL), prefill=Instance(kv_config=NIXL)),
    # NIXL rejects prefix caching on a hybrid decoder with a different TP.
    "pd-tp1-tp2": Deployment(
        Instance(tp=2, kv_config=NIXL, prefix_caching=False),
        prefill=Instance(kv_config=NIXL),
    ),
    "pd-offload": Deployment(
        Instance(kv_config=NIXL_OFFLOAD),
        prefill=Instance(kv_config=NIXL_OFFLOAD),
        offload=True,
    ),
    "pd-dcp2": Deployment(
        Instance(tp=2, dcp=2, kv_config=NIXL),
        prefill=Instance(tp=2, dcp=2, kv_config=NIXL),
    ),
}


KNOWN_FAILURES = {
    "pd-dcp2": "NIXL sets cp_kv_cache_interleave_size to the block size, "
    "but FlashInfer MLA DCP requires 1",
    "dcp2-offload": "CPU offload never hits with DCP on a hybrid model",
}


def _marks(name: str) -> list:
    num_gpus = DEPLOYMENTS[name].num_gpus
    marks = multi_gpu_marks(num_gpus=num_gpus) if num_gpus > 1 else []
    if name in KNOWN_FAILURES:
        marks.append(pytest.mark.xfail(reason=KNOWN_FAILURES[name], strict=True))
    return marks


def _block_size(url: str) -> int:
    response = requests.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name == "vllm:cache_config_info":
                return lcm(
                    *(
                        int(value)
                        for key in ("block_size", "mamba_block_size")
                        if (value := sample.labels.get(key, "None")) != "None"
                    )
                )
    raise AssertionError("missing vllm:cache_config_info metric")


def _complete(url: str, prompt: list[int], salt: str, **extra) -> dict:
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "logprobs": NUM_LOGPROBS,
        "return_tokens_as_token_ids": True,
        "cache_salt": salt,
        **extra,
    }
    response = requests.post(f"{url}/v1/completions", json=body, timeout=600)
    response.raise_for_status()
    return response.json()


def _pd_complete(
    prefill_url: str, decode_url: str, prompt: list[int], salt: str
) -> tuple[dict, dict]:
    remote_decode = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    prefill = _complete(
        prefill_url, prompt, salt, max_tokens=1, kv_transfer_params=remote_decode
    )
    decode = _complete(
        decode_url, prompt, salt, kv_transfer_params=prefill["kv_transfer_params"]
    )
    return prefill, decode


def _cached(response: dict) -> int:
    return response["usage"]["prompt_tokens_details"]["cached_tokens"]


def _token_id(token: str) -> int:
    return int(token.removeprefix("token_id:"))


def _tokens_text_logprobs(response: dict):
    choice = response["choices"][0]
    ids = [_token_id(t) for t in choice["logprobs"]["tokens"]]
    top = [
        {_token_id(t): lp for t, lp in d.items()}
        for d in choice["logprobs"]["top_logprobs"]
    ]
    return ids, choice["text"], top


def _reset_gpu_prefix_cache(url: str) -> None:
    response = requests.post(
        f"{url}/reset_prefix_cache", params={"reset_external": "false"}, timeout=60
    )
    response.raise_for_status()


@dataclass(frozen=True)
class Turn:
    prompt_len: int
    cached: int
    expected_cached: int
    recompute_cached: int
    output: tuple
    recompute: tuple

    @property
    def matches_recompute(self) -> bool:
        return self.output[0] == self.recompute[0]


def _expected_cached(prev_prompt_len: int, hit_unit: int) -> int:
    """A turn must reuse the previous prompt up to its last checkpoint."""
    return (prev_prompt_len - 1) // hit_unit * hit_unit if prev_prompt_len else 0


@contextlib.contextmanager
def _serve(
    deployment: Deployment, prefix_match_unit: int | None
) -> Iterator[list[RemoteOpenAIServer]]:
    with contextlib.ExitStack() as stack:
        # Sequential shutdown waits on memory still held by sibling servers.
        servers: list[RemoteOpenAIServer] = []
        stack.callback(RemoteOpenAIServer.shutdown_many, servers)
        next_gpu = 0
        for instance in deployment.instances:
            gpus = range(next_gpu, next_gpu + instance.tp)
            next_gpu += instance.tp
            env = {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
                "VLLM_SERVER_DEV_MODE": "1",
                "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(get_open_port()),
            }
            servers.append(
                RemoteOpenAIServer(
                    MODEL,
                    instance.args(prefix_match_unit),
                    env_dict=env,
                    max_wait_seconds=STARTUP_TIMEOUT,
                )
            )
        yield servers


def _run_conversation(
    deployment: Deployment,
    servers: list[RemoteOpenAIServer],
    prefix_match_unit: int | None,
) -> list[Turn]:
    # Under P/D the prefiller computes the prompt, so hits are read from it.
    compute_url, decode_url = servers[0].url_root, servers[-1].url_root
    block_size = _block_size(compute_url)
    hit_unit = prefix_match_unit or block_size
    rng = random.Random(0)

    def new_tokens(n: int) -> list[int]:
        return [rng.randint(1000, 150000) for _ in range(n)]

    prompt = new_tokens(FIRST_PROMPT_BLOCKS * block_size + UNALIGNED_TAIL)
    prev_len = 0
    turns = []
    for turn in range(NUM_TURNS):
        if deployment.offload:
            for server in servers:
                _reset_gpu_prefix_cache(server.url_root)
        if deployment.prefill is not None:
            computed, output = _pd_complete(compute_url, decode_url, prompt, "conv")
        else:
            computed = output = _complete(decode_url, prompt, "conv")
        recompute = _complete(decode_url, prompt, f"recompute-{turn}")
        turns.append(
            Turn(
                prompt_len=len(prompt),
                cached=_cached(computed),
                expected_cached=_expected_cached(prev_len, hit_unit),
                recompute_cached=_cached(recompute),
                output=_tokens_text_logprobs(output),
                recompute=_tokens_text_logprobs(recompute),
            )
        )
        prev_len = len(prompt)
        prompt = prompt + new_tokens(TURN_TOKENS)
    return turns


def _format(turns: list[Turn]) -> str:
    rows = ["turn  prompt  cached  expected  matches_recompute"]
    rows += [
        f"{i:>4}  {t.prompt_len:>6}  {t.cached:>6}  {t.expected_cached:>8}  "
        f"{t.matches_recompute}"
        for i, t in enumerate(turns)
    ]
    return "\n".join(rows)


@pytest.mark.parametrize("prefix_match_unit", [None, 128], ids=["block", "partial"])
@pytest.mark.parametrize(
    "name", [pytest.param(name, marks=_marks(name)) for name in DEPLOYMENTS]
)
def test_turns_reuse_prefix_and_match_recompute(
    name: str, prefix_match_unit: int | None
) -> None:
    deployment = DEPLOYMENTS[name]
    with _serve(deployment, prefix_match_unit) as servers:
        turns = _run_conversation(deployment, servers, prefix_match_unit)
    table = _format(turns)
    mode = "block" if prefix_match_unit is None else "partial"
    print(f"\n{name}-{mode}\n{table}")

    assert all(t.recompute_cached == 0 for t in turns), table
    assert all(t.cached >= t.expected_cached for t in turns), table
    # Dummy weights differ across TP sizes.
    if len({i.tp for i in deployment.instances}) == 1:
        check_logprobs_close(
            outputs_0_lst=[t.recompute for t in turns],
            outputs_1_lst=[t.output for t in turns],
            name_0="recompute",
            name_1=name,
        )
