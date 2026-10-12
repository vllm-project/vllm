# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 serving feature combinations on a 6-layer dummy-weight model.

Each deployment starts one or two servers and checks that completions, chat,
streaming, structured outputs and, where enabled, DSpark, image input, P/D
KV transfer and KV offload reloads all work together.
"""

import contextlib
import io
import json
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pybase64 as base64
import pytest
import requests
from PIL import Image
from prometheus_client.parser import text_string_to_metric_families

from tests.models.utils import check_logprobs_close
from tests.utils import RemoteOpenAIServer, multi_gpu_marks
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.network_utils import get_open_port

MODEL = "MiniMaxAI/MiniMax-M3"
DRAFT = "nvidia/MiniMax-M3-DSpark"
# Layers 0-2 are full attention with a dense MLP; the rest are MSA + MoE.
NUM_LAYERS = 6
LAYER_KINDS = [0, 0, 0, 1, 1, 1]
MAX_TOKENS = 8
NUM_LOGPROBS = 5
STARTUP_TIMEOUT = 1200

BASE_ARGS = [
    "--load-format",
    "dummy",
    "--block-size",
    "128",
    "--max-model-len",
    "8192",
    "--no-enable-flashinfer-autotune",
    "--enable-prompt-tokens-details",
    "--enable-prefix-caching",
    "--max-num-seqs",
    "8",
    "--reasoning-parser",
    "minimax_m3",
    "--tool-call-parser",
    "minimax_m3",
    "--enable-auto-tool-choice",
    # Dummy logits are near-uniform; without this a JSON grammar can loop on
    # whitespace until max_tokens.
    "--structured-outputs-config",
    json.dumps({"backend": "xgrammar", "disable_any_whitespace": True}),
]

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MiniMax-M3 MSA kernels require the SM100 family",
)

NIXL = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
NIXL_PUSH_P = {"kv_connector": "NixlPushConnector", "kv_role": "kv_producer"}
NIXL_PUSH_D = {"kv_connector": "NixlPushConnector", "kv_role": "kv_consumer"}
OFFLOADING = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"block_size": 128, "cpu_bytes_to_use": 8 << 30},
}
SIMPLE_OFFLOAD = {
    "kv_connector": "SimpleCPUOffloadConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use_per_rank": 8 << 30,
        "lazy_offload": False,
    },
}
NIXL_OFFLOADING = {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"connectors": [NIXL, OFFLOADING]},
}

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 9},
    },
    "required": ["answer", "score"],
    "additionalProperties": False,
}
COLORS = ["red", "green", "blue"]


@dataclass(frozen=True)
class Instance:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    dspark: bool = False
    image: bool = False
    hma: bool = True
    kv_config: dict | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def num_gpus(self) -> int:
        return self.tp * self.dp * self.pp

    def args(self, draft: str) -> list[str]:
        args = BASE_ARGS + ["--tensor-parallel-size", str(self.tp)]
        if self.dp > 1:
            args += ["--data-parallel-size", str(self.dp), "--enable-expert-parallel"]
        if self.pp > 1:
            args += ["--pipeline-parallel-size", str(self.pp)]
        if self.dspark:
            spec = {"method": "dspark", "model": draft, "num_speculative_tokens": 3}
            args += ["--speculative-config", json.dumps(spec)]
        if not self.image:
            args.append("--language-model-only")
        if not self.hma:
            args.append("--disable-hybrid-kv-cache-manager")
        if self.kv_config is not None:
            args += ["--kv-transfer-config", json.dumps(self.kv_config)]
        return args + list(self.extra_args)


@dataclass(frozen=True)
class Deployment:
    decode: Instance
    prefill: Instance | None = None
    # Reset GPU prefix caches before a resend so the hit must come from CPU.
    offload: bool = False

    @property
    def instances(self) -> tuple[Instance, ...]:
        return (self.prefill, self.decode) if self.prefill else (self.decode,)

    @property
    def num_gpus(self) -> int:
        return sum(i.num_gpus for i in self.instances)


# DSpark's sliding-window layers make the hybrid manager use mixed block sizes,
# which NixlConnector (#60791) and the CPU offload connectors do not handle, so
# connector + DSpark deployments turn it off; the *-hma cases track the failures.
NO_HMA_DSPARK = dict(dspark=True, hma=False)
DEPLOYMENTS = {
    "tp1": Deployment(Instance()),
    "tp1-dspark": Deployment(Instance(dspark=True)),
    "tp2-dspark": Deployment(Instance(tp=2, dspark=True)),
    "tp4-ep-dspark-image": Deployment(
        Instance(
            tp=4, dspark=True, image=True, extra_args=("--enable-expert-parallel",)
        )
    ),
    "dep4-dspark": Deployment(
        Instance(
            dp=4,
            dspark=True,
            extra_args=("--all2all-backend", "flashinfer_nvlink_one_sided"),
        )
    ),
    "offloading-dspark": Deployment(
        Instance(kv_config=OFFLOADING, **NO_HMA_DSPARK), offload=True
    ),
    "simple-offload-dspark": Deployment(
        Instance(kv_config=SIMPLE_OFFLOAD, **NO_HMA_DSPARK), offload=True
    ),
    "simple-offload-dspark-hma": Deployment(
        Instance(dspark=True, kv_config=SIMPLE_OFFLOAD), offload=True
    ),
    "pd-dspark-image": Deployment(
        Instance(image=True, kv_config=NIXL, **NO_HMA_DSPARK),
        prefill=Instance(image=True, kv_config=NIXL, **NO_HMA_DSPARK),
    ),
    "pd-push-dspark": Deployment(
        Instance(kv_config=NIXL_PUSH_D, **NO_HMA_DSPARK),
        prefill=Instance(kv_config=NIXL_PUSH_P, **NO_HMA_DSPARK),
    ),
    "pd-tp2-dep2-dspark": Deployment(
        Instance(dp=2, kv_config=NIXL, **NO_HMA_DSPARK),
        prefill=Instance(tp=2, kv_config=NIXL, **NO_HMA_DSPARK),
    ),
    "pd-offloading-dspark": Deployment(
        Instance(kv_config=NIXL_OFFLOADING, **NO_HMA_DSPARK),
        prefill=Instance(kv_config=NIXL_OFFLOADING, **NO_HMA_DSPARK),
        offload=True,
    ),
    "pd-dspark-hma": Deployment(
        Instance(kv_config=NIXL, dspark=True),
        prefill=Instance(kv_config=NIXL, dspark=True),
    ),
    "pp2": Deployment(Instance(pp=2)),
}

KNOWN_FAILURES = {
    "simple-offload-dspark-hma": "CPU offload never reloads with HMA + DSpark (#60791)",
    "pd-dspark-hma": "NIXL cannot split KV regions with HMA + DSpark (#60791)",
    "pp2": "PP stages disagree on supported KV cache layouts (#54710)",
}


def _case(name: str):
    marks = []
    if (num_gpus := DEPLOYMENTS[name].num_gpus) > 1:
        marks += multi_gpu_marks(num_gpus=num_gpus)
    if reason := KNOWN_FAILURES.get(name):
        marks.append(pytest.mark.xfail(reason=reason, strict=True))
    return pytest.param(name, marks=marks, id=name)


@pytest.fixture(scope="module")
def model_dirs(tmp_path_factory) -> tuple[str, str]:
    """Config-only copies of MODEL and DRAFT, cut to NUM_LAYERS target layers."""
    root = tmp_path_factory.mktemp("minimax_m3")
    target, draft = root / "target", root / "draft"
    src = Path(
        hf_api().snapshot_download(
            MODEL,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "chat_template.jinja",
                "tokenizer*.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    )
    # Copy, since the snapshot files are symlinks into the HF cache.
    shutil.copytree(src, target, symlinks=False)
    config = json.loads((target / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = NUM_LAYERS
    text["moe_layer_freq"] = LAYER_KINDS
    text["num_mtp_modules"] = 1
    sparse = text["sparse_attention_config"]
    sparse["sparse_attention_freq"] = LAYER_KINDS
    sparse["sparse_disable_index_value"] = LAYER_KINDS
    (target / "config.json").write_text(json.dumps(config))

    draft_src = Path(hf_api().snapshot_download(DRAFT, allow_patterns=["config.json"]))
    draft_config = json.loads((draft_src / "config.json").read_text())
    draft_config["dflash_config"]["target_layer_ids"] = list(range(NUM_LAYERS))
    draft_config["num_target_layers"] = NUM_LAYERS
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(draft_config))
    return str(target), str(draft)


@contextlib.contextmanager
def _proxy(prefill_url: str, decode_url: str) -> Iterator[str]:
    port = get_open_port()
    script = Path(__file__).parents[2] / (
        "v1/kv_connector/nixl_integration/toy_proxy_server.py"
    )
    args = [sys.executable, str(script), "--port", str(port)]
    for role, url in (("prefiller", prefill_url), ("decoder", decode_url)):
        host, url_port = url.removeprefix("http://").rsplit(":", 1)
        args += [f"--{role}-hosts", host, f"--{role}-ports", url_port]
    proc = subprocess.Popen(args, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, "toy proxy exited"
            with contextlib.suppress(requests.ConnectionError):
                if requests.get(f"{url}/healthcheck", timeout=5).ok:
                    break
            assert time.monotonic() < deadline, "toy proxy did not start"
            time.sleep(0.5)
        yield url
    finally:
        proc.kill()
        proc.wait()


@contextlib.contextmanager
def _serve(
    deployment: Deployment, model_dirs: tuple[str, str]
) -> Iterator[list[RemoteOpenAIServer]]:
    target, draft = model_dirs
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
                env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(get_open_port())
            servers.append(
                RemoteOpenAIServer(
                    target,
                    instance.args(draft),
                    env_dict=env,
                    max_wait_seconds=STARTUP_TIMEOUT,
                )
            )
        yield servers


def _metric(url: str, name: str, **labels: str) -> float:
    response = requests.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    return sum(
        sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(k) == v for k, v in labels.items())
    )


def _post(url: str, path: str, body: dict) -> dict:
    response = requests.post(f"{url}/v1/{path}", json=body, timeout=600)
    response.raise_for_status()
    return response.json()


def _complete(url: str, model: str, prompt: list[int], salt: str, **extra) -> dict:
    return _post(
        url,
        "completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "logprobs": NUM_LOGPROBS,
            "return_tokens_as_token_ids": True,
            "cache_salt": salt,
            **extra,
        },
    )


def _chat(url: str, model: str, content, **extra) -> dict:
    return _post(
        url,
        "chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 32,
            "temperature": 0,
            **extra,
        },
    )


def _tokens_text_logprobs(response: dict):
    def token_id(token: str) -> int:
        return int(token.removeprefix("token_id:"))

    choice = response["choices"][0]
    ids = [token_id(t) for t in choice["logprobs"]["tokens"]]
    top = [
        {token_id(t): lp for t, lp in d.items()}
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


def _check_structured_outputs(url: str, model: str) -> None:
    """Bounded grammars must finish and parse even with dummy weights."""
    # With thinking disabled the grammar applies from the first output token.
    no_think = {"thinking_mode": "disabled"}
    schema = {"name": "answer", "schema": PERSON_SCHEMA}
    response = _chat(
        url,
        model,
        "Answer as JSON.",
        response_format={"type": "json_schema", "json_schema": schema},
        chat_template_kwargs=no_think,
    )
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop", choice
    parsed = json.loads(choice["message"]["content"])
    assert parsed["answer"] in ("yes", "no") and 0 <= parsed["score"] <= 9, parsed

    response = _chat(
        url,
        model,
        "Pick a color.",
        structured_outputs={"choice": COLORS},
        chat_template_kwargs=no_think,
    )
    assert response["choices"][0]["message"]["content"] in COLORS, response


def _check_chat(url: str, model: str, image: bool) -> None:
    response = _chat(url, model, "Hello")
    assert response["usage"]["completion_tokens"] > 0, response

    stream = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Count to five."}],
            "max_tokens": 16,
            "temperature": 0,
            "stream": True,
        },
        stream=True,
        timeout=600,
    )
    stream.raise_for_status()
    chunks = [line for line in stream.iter_lines() if line.startswith(b"data: {")]
    assert chunks, "empty stream"

    if image:
        buf = io.BytesIO()
        Image.new("RGB", (448, 448), (200, 40, 40)).save(buf, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": "What color is this image?"},
        ]
        response = _chat(url, model, content)
        assert response["usage"]["prompt_tokens"] > 100, response


@pytest.mark.parametrize("name", [_case(name) for name in DEPLOYMENTS])
def test_feature_combination(name: str, model_dirs: tuple[str, str]) -> None:
    deployment = DEPLOYMENTS[name]
    decode = deployment.decode
    rng = random.Random(0)
    prompt = [rng.randint(1000, 150000) for _ in range(3 * 128 + 37)]
    with _serve(deployment, model_dirs) as servers:
        # Under P/D the prefiller computes prompts, so offload hits are read there.
        compute_url, decode_url = servers[0].url_root, servers[-1].url_root
        model = model_dirs[0]
        if deployment.prefill is None:
            proxy = contextlib.nullcontext(decode_url)
        else:
            proxy = _proxy(compute_url, decode_url)
        with proxy as url:
            external_before = _metric(
                decode_url,
                "vllm:prompt_tokens_by_source_total",
                source="external_kv_transfer",
            )
            output = _complete(url, model, prompt, "first")
            # A local recompute on the decoder is the reference output.
            recompute = _complete(decode_url, model, prompt, "recompute")
            if deployment.offload:
                _complete(compute_url, model, prompt, "offload")
                _reset_gpu_prefix_cache(compute_url)
                resend = _complete(compute_url, model, prompt, "offload")
                cached = resend["usage"]["prompt_tokens_details"]["cached_tokens"]
                assert cached > 0, resend["usage"]

            with ThreadPoolExecutor(8) as pool:
                prompts = [
                    [rng.randint(1000, 150000) for _ in range(rng.randint(64, 2048))]
                    for _ in range(8)
                ]
                for response in pool.map(
                    lambda p: _complete(url, model, p, "concurrent"), prompts
                ):
                    assert response["usage"]["completion_tokens"] > 0, response

            _check_chat(url, model, decode.image)
            _check_structured_outputs(url, model)

        if deployment.prefill is not None:
            external = _metric(
                decode_url,
                "vllm:prompt_tokens_by_source_total",
                source="external_kv_transfer",
            )
            assert external > external_before, "decoder received no KV transfer"
        if decode.dspark:
            drafts = _metric(decode_url, "vllm:spec_decode_num_drafts_total")
            assert drafts > 0, "DSpark proposed no drafts"

    # Dummy weights differ across TP sizes.
    if len({(i.tp, i.dp, i.pp) for i in deployment.instances}) == 1:
        check_logprobs_close(
            outputs_0_lst=[_tokens_text_logprobs(recompute)],
            outputs_1_lst=[_tokens_text_logprobs(output)],
            name_0="recompute",
            name_1=name,
        )
