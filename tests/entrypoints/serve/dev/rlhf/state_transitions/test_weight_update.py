# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real HTTP weight transfer and repeated RL steps with a numerical oracle.

IPC uses one GPU; NCCL needs a second GPU for the trainer. Both runners are
covered. The test builds B from the same checkpoint, so no optional alternate
checkpoint can silently skip the transfer. Worker ordering errors and model/
quantization reload matrices remain in their worker and model-loader suites.
Cache invalidation is explicitly owned by pause(clear_cache=True), matching
the current API; this does not claim finish_weight_update invalidates caches.
"""

import pytest
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.entrypoints.serve.dev.rlhf.conftest import (
    cached_tokens,
    pause,
    resume,
    server,
    sleep,
    wake,
)
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
from vllm.utils.network_utils import get_open_port

MODEL = "facebook/opt-125m"
PROMPT = "Paris is the capital of France. Berlin is the capital of Germany. " * 8


class VersionedClient(HTTPVLLMWeightSyncClient):
    """Pass the round's version through the real trainer finish handshake."""

    version = "default"

    def finish_weight_update(self, weight_version=None):
        super().finish_weight_update(self.version)


@pytest.fixture(params=[False, True], ids=["MRV1", "MRV2"])
def use_v2(request):
    return request.param


@pytest.fixture(params=["ipc", pytest.param("nccl", marks=pytest.mark.distributed)])
def backend(request, num_gpus_available):
    if not torch.cuda.is_available():
        pytest.skip("The IPC/NCCL transfer backends require CUDA")
    if request.param == "nccl" and num_gpus_available < 2:
        pytest.skip("NCCL needs a separate trainer GPU")
    return request.param


def completion(url):
    response = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "m",
            "prompt": PROMPT,
            "max_tokens": 8,
            "temperature": 0,
            "logprobs": 1,
            "return_token_ids": True,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def assert_matches_trainer(response, model, tokenizer):
    """Compare the same generated token positions, not free-running HF text."""
    choice = response["choices"][0]
    generated = choice["token_ids"]
    assert generated
    prompt = tokenizer.encode(PROMPT)
    ids = torch.tensor([prompt + generated], device=model.device)
    with torch.inference_mode():
        logits = model(ids).logits[0, len(prompt) - 1 : -1].float()
        targets = torch.tensor(generated, device=model.device)
        expected = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    actual = choice["logprobs"]["token_logprobs"]
    assert len(actual) == len(generated)
    assert torch.isfinite(expected).all()
    assert actual == pytest.approx(expected.cpu().tolist(), abs=0.05, rel=0)


def test_transfer_commits_weights_and_version_across_rl_steps(
    backend, use_v2, monkeypatch
):
    """Pause-only B update followed by level-2 restoration to A both match HF."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    device = 1 if backend == "nccl" else 0
    previous_device = torch.accelerator.current_device_index()
    torch.accelerator.set_device_index(device)
    model = head = original = trainer = None
    try:
        model = (
            AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
            .to(f"cuda:{device}")
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # Alter a real checkpoint tensor in place; ModuleSource replays it each round.
        head = model.get_output_embeddings().weight
        original = head.detach().clone()
        with server(
            model=MODEL,
            env_dict={"VLLM_USE_V2_MODEL_RUNNER": str(int(use_v2))},
            extra_args=[
                "--device-ids",
                "0",
                "--gpu-memory-utilization",
                "0.4",
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
            ],
            weight_transfer_config={"backend": backend},
        ) as url:
            baseline = completion(url)
            assert_matches_trainer(baseline, model, tokenizer)
            assert cached_tokens(completion(url)) > 0
            init_info = (
                IPCTrainerInitInfo(rank=0, packed=False)
                if backend == "ipc"
                else NCCLTrainerInitInfo(
                    rank=0,
                    master_address="127.0.0.1",
                    master_port=get_open_port(),
                    world_size=2,
                    packed=False,
                )
            )
            client = VersionedClient(url)
            trainer = WeightTransferTrainerFactory.trainer_init(
                init_info=init_info, client=client, source=ModuleSource(model)
            )
            try:
                for version, deep_sleep in [("B", False), ("A-restored", True)]:
                    with torch.no_grad():
                        head.copy_(original * (0.5 if version == "B" else 1.0))
                    assert pause(url, clear_cache=True) == 200
                    if deep_sleep:
                        assert sleep(url, level=2) == 200
                        assert wake(url, tags=["weights"]) == 200
                    client.version = version
                    trainer.send_weights()
                    info = requests.get(f"{url}/weight_info", timeout=10)
                    info.raise_for_status()
                    assert info.json() == {"weight_version": version}
                    if deep_sleep:
                        assert wake(url, tags=["kv_cache"]) == 200
                    assert resume(url) == 200
                    result = completion(url)
                    assert cached_tokens(result) == 0
                    assert_matches_trainer(result, model, tokenizer)
                    if version == "A-restored":
                        assert (
                            result["choices"][0]["token_ids"]
                            == baseline["choices"][0]["token_ids"]
                        )
                    assert cached_tokens(completion(url)) > 0
            finally:
                trainer.shutdown()
    finally:
        del trainer, head, original, model
        torch.accelerator.empty_cache()
        torch.accelerator.set_device_index(previous_device)
