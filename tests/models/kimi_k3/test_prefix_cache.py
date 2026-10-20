# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 multi-turn prefix cache reuse with KV offload, P/D and DCP."""

import contextlib
import json
import os
import random
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from math import lcm
from uuid import uuid4

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.models.utils import check_logprobs_close
from tests.utils import RemoteOpenAIServer, multi_gpu_marks
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

# Official Kimi-K3 config with 8 layers and no weights.
MODEL = "riverclouds/Kimi-K3-8L-dummy"
# The recipe's DSpark draft, reading target layers that exist in MODEL.
DRAFT = "riverclouds/Kimi-K3-DSpark-for-8L-dummy"

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

SPEC_CONFIG = {
    "method": "dspark",
    "model": DRAFT,
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "rejection_sample_method": "block",
}

NIXL = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
MOONCAKE = {
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "load_async": True,
        "lookup_async": True,
        "enable_cross_layers_blocks": False,
    },
}
# SimpleCPUOffloadConnector as in the Kimi-K3 recipe.
OFFLOAD = {
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use_per_rank": 32 << 30,
        "lazy_offload": False,
    },
}
NIXL_OFFLOAD = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"connectors": [NIXL, OFFLOAD]},
}


@dataclass(frozen=True)
class Mode:
    # A finer prefix_match_unit enables partial hits inside a Mamba block.
    prefix_match_unit: int | None
    spec: bool

    @property
    def name(self) -> str:
        name = "block" if self.prefix_match_unit is None else "partial"
        return f"{name}-dspark" if self.spec else name


MODES = [Mode(unit, spec) for spec in (False, True) for unit in (None, 128)]


@dataclass(frozen=True)
class Instance:
    tp: int = 1
    dp: int = 1
    dcp: int = 1
    kv_config: dict | None = None
    prefix_caching: bool = True

    @property
    def num_gpus(self) -> int:
        return self.tp * self.dp

    def args(self, mode: Mode) -> list[str]:
        args = BASE_ARGS + ["--tensor-parallel-size", str(self.tp)]
        if mode.spec:
            args += ["--speculative-config", json.dumps(SPEC_CONFIG)]
        if self.dp > 1:
            args += ["--data-parallel-size", str(self.dp), "--enable-expert-parallel"]
        if self.dcp > 1:
            args += ["--decode-context-parallel-size", str(self.dcp)]
        if self.kv_config is not None:
            args += ["--kv-transfer-config", json.dumps(self.kv_config)]
        if not self.prefix_caching:
            return args + ["--no-enable-prefix-caching"]
        args.append("--enable-prefix-caching")
        if mode.prefix_match_unit is not None:
            args += ["--prefix-match-unit", str(mode.prefix_match_unit)]
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
        return sum(i.num_gpus for i in self.instances)


DEPLOYMENTS = {
    "plain": Deployment(Instance()),
    "offload": Deployment(Instance(kv_config=OFFLOAD), offload=True),
    "dcp2": Deployment(Instance(tp=2, dcp=2)),
    "dcp2-offload": Deployment(Instance(tp=2, dcp=2, kv_config=OFFLOAD), offload=True),
    "pd": Deployment(Instance(kv_config=NIXL), prefill=Instance(kv_config=NIXL)),
    # NIXL rejects prefix caching on a hybrid decoder with a different TP.
    "pd-tp2-dep2": Deployment(
        Instance(dp=2, kv_config=NIXL, prefix_caching=False),
        prefill=Instance(tp=2, kv_config=NIXL),
    ),
    "pd-offload": Deployment(
        Instance(kv_config=NIXL_OFFLOAD),
        prefill=Instance(kv_config=NIXL_OFFLOAD),
        offload=True,
    ),
}


def _known_failure(name: str, mode: Mode) -> str | None:
    deployment = DEPLOYMENTS[name]
    if mode.spec and deployment.decode.dcp > 1:
        return "FlashInfer MLA DCP decode breaks on ragged spec batches (#59392)"
    return None


def _case(name: str, mode: Mode):
    marks = []
    if (num_gpus := DEPLOYMENTS[name].num_gpus) > 1:
        marks += multi_gpu_marks(num_gpus=num_gpus)
    if reason := _known_failure(name, mode):
        marks.append(pytest.mark.xfail(reason=reason, strict=True))
    return pytest.param(name, mode, marks=marks, id=f"{name}-{mode.name}")


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
    deadline = time.monotonic() + 60
    while True:
        response = requests.post(
            f"{url}/reset_prefix_cache", params={"reset_external": "false"}, timeout=60
        )
        response.raise_for_status()
        if response.json()["success"]:
            return
        assert time.monotonic() < deadline, "GPU cache reset timed out"
        time.sleep(0.1)


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


def _expected_cached(prev_prompt_len: int, hit_unit: int, spec: bool) -> int:
    """A turn must reuse the previous prompt up to its last checkpoint.

    EAGLE-style drafts drop the trailing unit, so the checkpoint is one earlier.
    """
    if not prev_prompt_len:
        return 0
    checkpoint = (prev_prompt_len - 1) // hit_unit * hit_unit
    return max(checkpoint - hit_unit, 0) if spec else checkpoint


@contextlib.contextmanager
def _serve(deployment: Deployment, mode: Mode) -> Iterator[list[RemoteOpenAIServer]]:
    with contextlib.ExitStack() as stack:
        # Sequential shutdown waits on memory still held by sibling servers.
        servers: list[RemoteOpenAIServer] = []
        stack.callback(RemoteOpenAIServer.shutdown_many, servers)
        next_gpu = 0
        for instance in deployment.instances:
            env = {}
            if deployment.offload:
                # Enables /reset_prefix_cache.
                env["VLLM_SERVER_DEV_MODE"] = "1"
            if deployment.prefill is not None:
                gpus = range(next_gpu, next_gpu + instance.num_gpus)
                next_gpu += instance.num_gpus
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
                env["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
                env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(get_open_port())
            servers.append(
                RemoteOpenAIServer(
                    MODEL,
                    instance.args(mode),
                    env_dict=env,
                    max_wait_seconds=STARTUP_TIMEOUT,
                )
            )
        yield servers


def _run_conversation(
    deployment: Deployment,
    servers: list[RemoteOpenAIServer],
    mode: Mode,
) -> list[Turn]:
    # Under P/D the prefiller computes the prompt, so hits are read from it.
    compute_url, decode_url = servers[0].url_root, servers[-1].url_root
    block_size = _block_size(compute_url)
    hit_unit = mode.prefix_match_unit or block_size
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
                expected_cached=_expected_cached(prev_len, hit_unit, mode.spec),
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


def _check_resend_reuses_prompt_checkpoint(
    deployment: Deployment, prompt_len: int
) -> None:
    """Resends reuse the Mamba checkpoint and its companion EAGLE attention proof."""
    mode = Mode(128, True)
    salt = str(uuid4())
    rng = random.Random(0)
    prompt = [rng.randint(1000, 150000) for _ in range(prompt_len)]
    with _serve(deployment, mode) as servers:
        url = servers[0].url_root
        cold = _complete(url, prompt, salt, max_tokens=1)
        if deployment.offload:
            _reset_gpu_prefix_cache(url)
        resend = _complete(url, prompt, salt, max_tokens=1)
        recompute = _complete(url, prompt, f"{salt}-recompute", max_tokens=1)

    # EAGLE's dropped hash unit already leaves tokens to recompute on resend.
    expected = (prompt_len // 128 - 1) * 128
    actual = _cached(resend)
    assert _cached(cold) == _cached(recompute) == 0
    assert actual == expected, f"{prompt_len=}: {actual=}, {expected=}"
    check_logprobs_close(
        outputs_0_lst=[_tokens_text_logprobs(recompute)],
        outputs_1_lst=[_tokens_text_logprobs(resend)],
        name_0="recompute",
        name_1="resend",
    )


@pytest.mark.parametrize(
    "prompt_len",
    [
        # Blocks=6144, PMU=128. P=7040 is PMU-aligned but not block-aligned:
        # ensure lookup reads the cached attention proof at 7040 despite
        # the final hit limit of 7039, then applies EAGLE's 128-token drop.
        pytest.param(7040, id="eagle-proof-scan"),
        # Blocks=6144, PMU=128. P=7296 is an exact PMU multiple:
        # ensure the Mamba checkpoint is 7168, not 7040 from applying both
        # the P - 1 cap and EAGLE's 128-token drop.
        pytest.param(7296, id="eagle-double-cap"),
    ],
)
def test_dspark_exact_resend_reuses_prompt_checkpoint(prompt_len: int) -> None:
    _check_resend_reuses_prompt_checkpoint(DEPLOYMENTS["plain"], prompt_len)


@pytest.fixture
def mooncake_store(tmp_path, monkeypatch) -> Iterator[None]:
    """Run a Mooncake master and point the servers' store config at it."""
    if shutil.which("mooncake_master") is None:
        pytest.skip("mooncake_master is not installed")
    port = get_open_port()
    log_path = tmp_path / "mooncake_master.log"
    with open(log_path, "w") as log:
        master = subprocess.Popen(
            [
                "mooncake_master",
                f"--port={port}",
                f"--metrics_port={get_open_port()}",
                "--enable_metric_reporting=false",
                "--default_kv_lease_ttl=60000",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 60
        while True:
            assert master.poll() is None, log_path.read_text()
            with (
                contextlib.suppress(OSError),
                socket.create_connection(("127.0.0.1", port)),
            ):
                break
            assert time.monotonic() < deadline, "Mooncake master did not start"
            time.sleep(0.1)
        config_path = tmp_path / "mooncake.json"
        config_path.write_text(
            json.dumps(
                {
                    "metadata_server": "P2PHANDSHAKE",
                    "master_server_address": f"127.0.0.1:{port}",
                    "protocol": "rdma",
                    "device_name": os.getenv("MOONCAKE_DEVICE_NAME", "mlx5_12"),
                    "global_segment_size": "4GB",
                    "local_buffer_size": "1GB",
                }
            )
        )
        monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
        monkeypatch.setenv("MC_MAX_MR_SIZE", str(4 << 30))
        yield
    finally:
        master.kill()
        master.wait()


@pytest.mark.usefixtures("mooncake_store")
def test_mooncake_resend_reuses_prompt_checkpoint() -> None:
    """Blocks=6144, PMU=128. P=7449 needs Mamba state at 7296 and attention proof
    at 7424, both beyond the normal Mooncake save boundary of 6144: ensure they
    remain reusable from the store after a GPU cache reset."""
    _check_resend_reuses_prompt_checkpoint(
        Deployment(Instance(kv_config=MOONCAKE), offload=True), 7449
    )


@pytest.mark.parametrize(
    "name, mode", [_case(name, mode) for name in DEPLOYMENTS for mode in MODES]
)
def test_turns_reuse_prefix_and_match_recompute(name: str, mode: Mode) -> None:
    deployment = DEPLOYMENTS[name]
    with _serve(deployment, mode) as servers:
        turns = _run_conversation(deployment, servers, mode)
    table = _format(turns)
    print(f"\n{name}-{mode.name}\n{table}")

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
