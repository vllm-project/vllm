# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import pytest
import regex as re
import requests
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    TokenizersBackend,
)

from tests.entrypoints.rl.conftest import (
    MODEL_NAME,
    gen,
    health,
    ok,
    server,
)
from tests.utils import RemoteOpenAIServer, multi_gpu_test
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
from vllm.platforms import current_platform


def weight_checker(
    url: str, action: str, baseline=None, model=None
) -> requests.Response:
    body = {"action": action}
    if baseline is not None:
        body["baseline"] = baseline
    if model is not None:
        body["model"] = model
    return requests.post(f"{url}/weight_checker", json=body, timeout=300)


def checksums(url: str, model=None) -> dict[str, str]:
    response = weight_checker(url, "checksum", model=model)
    response.raise_for_status()
    return response.json()["checksums"]


def ranks(keys) -> set[str]:
    return {key.rsplit(":", 1)[0] for key in keys}


def reset_reload_and_compare(url: str) -> None:
    baseline = checksums(url)
    try:
        weight_checker(url, "reset").raise_for_status()
        comparison = weight_checker(url, "compare", baseline).json()
        assert ranks(comparison["mismatches"]) == ranks(baseline)
    finally:
        requests.post(
            f"{url}/collective_rpc", json={"method": "reload_weights"}, timeout=300
        ).raise_for_status()
    assert weight_checker(url, "compare", baseline).json() == {
        "match": True,
        "mismatches": [],
    }


@pytest.fixture(scope="class")
def server_url():
    with server() as url:
        yield url


class TestSingleGPU:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"action": "frobnicate"},
            {"action": "compare"},
            {"action": "checksum", "model": "model"},
            [],
            None,
        ],
    )
    def test_invalid_request_returns_400(self, server_url, payload):
        response = requests.post(
            f"{server_url}/weight_checker", json=payload, timeout=10
        )
        assert response.status_code == 400, response.text
        assert health(server_url) == 200

    def test_null_model_reset_does_not_change_weights(self, server_url):
        baseline = checksums(server_url)
        response = requests.post(
            f"{server_url}/weight_checker",
            json={"action": "reset", "model": None},
            timeout=300,
        )
        assert response.status_code == 400, response.text
        assert checksums(server_url) == baseline

    @pytest.mark.parametrize("model", ["draft", "both"])
    def test_missing_draft_rejects_all_actions_without_resetting_target(
        self, server_url, model
    ):
        target = checksums(server_url)
        for action in ("checksum", "reset", "compare"):
            response = weight_checker(server_url, action, baseline={}, model=model)
            assert response.status_code == 409, response.text
            error = response.json()["error"]
            assert error["type"] == "Conflict" and error["code"] == 409
            assert error["param"] is None
            assert "resident speculative draft model" in error["message"]
            if action == "reset":
                assert "other workers may have reset" in error["message"]
        assert checksums(server_url) == target

    def test_checksum_is_stable_and_compare_reports_differences(self, server_url):
        baseline = checksums(server_url)
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", digest) for digest in baseline.values()
        )
        assert checksums(server_url) == baseline

        changed, missing, *_ = baseline
        altered = {**baseline, changed: "0" * 64}
        del altered[missing]
        assert weight_checker(server_url, "compare", altered).json() == {
            "match": False,
            "mismatches": sorted([changed, missing]),
        }

    def test_reset_changes_weights_and_reload_restores_them(self, server_url):
        reset_reload_and_compare(server_url)
        assert ok(gen(server_url))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="IPC weight transfer uses CUDA IPC."
)
def test_ipc_weight_transfer_restores_reset_weights():
    """The documented RL flow: reset, then transfer the checkpoint back."""
    if current_platform.is_xpu():
        pytest.skip("IPC weight transfer backend uses CUDA IPC handles")
    args = ["--weight-transfer-config", '{"backend": "ipc"}']
    with (
        patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}),
        server(extra_args=args, port=8772) as url,
    ):
        baseline = checksums(url)
        weight_checker(url, "reset").raise_for_status()
        trainer_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16
        ).cuda()
        client = HTTPVLLMWeightSyncClient(url)
        # send_weights() finishes the update itself; this makes it verify too.
        client.finish_weight_update = partial(  # type: ignore[method-assign]
            client.finish_weight_update, baseline=baseline
        )
        trainer = WeightTransferTrainerFactory.trainer_init(
            IPCTrainerInitInfo(rank=0),
            client=client,
            source=ModuleSource(trainer_model),
        )
        trainer.send_weights()

        with torch.no_grad():
            trainer_model.model.norm.weight.add_(1)
        with pytest.raises(RuntimeError, match="model.norm.weight"):
            trainer.send_weights()


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "args, expected",
    [
        (["--data-parallel-size", "2"], {"dp0:pp0:pcp0:tp0", "dp1:pp0:pcp0:tp0"}),
        (["--tensor-parallel-size", "2"], {"dp0:pp0:pcp0:tp0", "dp0:pp0:pcp0:tp1"}),
    ],
)
def test_checksum_and_reset_cover_every_worker(args, expected):
    args = [*draft_args(), *args, "--weight-transfer-config", '{"backend": "nccl"}']
    with server(extra_args=args, port=8771, timeout=600) as url:
        requests.post(f"{url}/pause?mode=abort", timeout=30).raise_for_status()
        assert ranks(checksums(url)) == expected
        reset_reload_and_compare(url)

        expected_both = {
            f"{model}:{rank}" for model in ("target", "draft") for rank in expected
        }
        baseline = checksums(url, "both")
        assert ranks(baseline) == expected_both
        weight_checker(url, "reset", model="both").raise_for_status()
        comparison = weight_checker(url, "compare", baseline, model="both")
        comparison.raise_for_status()
        result = comparison.json()
        assert result["match"] is False
        assert ranks(result["mismatches"]) == expected_both


def test_weight_checker_rejects_a_draft_without_weight_parameters():
    args = [
        "--speculative-config",
        json.dumps(
            {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {"eagle_aux_hidden_state_layer_ids": [0, 1]}
                },
            }
        ),
    ]
    with server(extra_args=args) as url:
        target = checksums(url)
        for action, model in (
            ("checksum", "draft"),
            ("compare", "draft"),
            ("reset", "draft"),
            ("reset", "both"),
        ):
            response = weight_checker(url, action, baseline={}, model=model)
            assert response.status_code == 409, response.text
            message = response.json()["error"]["message"]
            assert "draft model with nonempty weights" in message
        assert checksums(url) == target


def draft_args():
    return [
        "--speculative-config",
        json.dumps(
            {
                "method": "draft_model",
                "model": MODEL_NAME,
                "num_speculative_tokens": 2,
            }
        ),
    ]


def test_draft_reset_requires_a_supported_weight_transfer_backend():
    with server(extra_args=draft_args()) as url:
        requests.post(f"{url}/pause?mode=abort", timeout=30).raise_for_status()
        baseline = checksums(url, "both")
        for model in ("draft", "both"):
            response = weight_checker(url, "reset", model=model)
            assert response.status_code == 409, response.text
            message = response.json()["error"]["message"]
            assert "backend that supports draft weight updates" in message
        assert checksums(url, "both") == baseline


class DraftWeightSyncClient(HTTPVLLMWeightSyncClient):
    def start_weight_update(self) -> None:
        self._post("start_draft_weight_update")


def send_ipc_weights(client: HTTPVLLMWeightSyncClient, module: nn.Module) -> None:
    WeightTransferTrainerFactory.trainer_init(
        IPCTrainerInitInfo(rank=0), client=client, source=ModuleSource(module.cuda())
    ).send_weights()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="IPC weight transfer uses CUDA IPC."
)
def test_ipc_transfer_restores_target_and_draft_weights():
    args = [*draft_args(), "--weight-transfer-config", '{"backend": "ipc"}']
    with (
        patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}),
        server(extra_args=args, port=8772) as url,
    ):
        requests.post(f"{url}/pause?mode=abort", timeout=30).raise_for_status()
        target = checksums(url, "target")
        draft = checksums(url, "draft")
        both = {f"target:{key}": digest for key, digest in target.items()}
        both.update(draft)
        assert target and draft
        assert all(key.startswith("draft:") for key in draft)
        assert checksums(url) == target
        assert checksums(url, "both") == both
        comparison = weight_checker(url, "compare", draft)
        comparison.raise_for_status()
        assert comparison.json() == {
            "match": False,
            "mismatches": sorted(target.keys() | draft.keys()),
        }

        class ResetCheckingDraftClient(DraftWeightSyncClient):
            def start_weight_update(self) -> None:
                super().start_weight_update()
                for selection in ("target", "draft", "both"):
                    response = weight_checker(url, "reset", model=selection)
                    assert response.status_code == 409, response.text
                    assert "while a weight update is active" in response.text

        clients = {
            "target": HTTPVLLMWeightSyncClient(url),
            "draft": ResetCheckingDraftClient(url),
        }

        trainer_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16
        )
        for model, baseline, restored_models in (
            ("target", target, ("target",)),
            ("draft", draft, ("draft",)),
            ("both", both, ("target", "draft")),
        ):
            weight_checker(url, "reset", model=model).raise_for_status()
            mismatch = weight_checker(url, "compare", baseline, model=model)
            mismatch.raise_for_status()
            assert mismatch.json()["match"] is False
            assert ranks(mismatch.json()["mismatches"]) == ranks(baseline)
            if model == "target":
                assert checksums(url, "draft") == draft
            elif model == "draft":
                assert checksums(url, "target") == target

            for restored_model in restored_models:
                send_ipc_weights(clients[restored_model], trainer_model)

            comparison = weight_checker(url, "compare", baseline, model=model)
            comparison.raise_for_status()
            assert comparison.json() == {"match": True, "mismatches": []}
            if model == "draft":
                assert checksums(url) == target
        requests.post(f"{url}/resume", timeout=30).raise_for_status()
        assert ok(gen(url))


@pytest.fixture
def eagle_checkpoints(tmp_path):
    """Small Llama target and an EAGLE checkpoint with no embed or head weights."""
    target_path = tmp_path / "target"
    draft_path = tmp_path / "eagle-draft"
    draft_path.mkdir()
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        target = LlamaForCausalLM(config).to(torch.bfloat16)
        target.save_pretrained(target_path)
        draft_config = LlamaConfig.from_pretrained(target_path, num_hidden_layers=1)
        draft = LlamaModel(draft_config)
        del draft.embed_tokens, draft.norm
        draft.layers[0].input_layernorm = nn.Identity()
        draft.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        draft.to(torch.bfloat16)
    draft_config.save_pretrained(draft_path)
    save_file(draft.state_dict(), draft_path / "model.safetensors")

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({f"token{i}": i for i in range(4, config.vocab_size)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    TokenizersBackend(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    ).save_pretrained(target_path)
    return target_path, draft_path, target, draft


@contextmanager
def paused_eagle_server(eagle_checkpoints, pp_size):
    target_path, draft_path, _, _ = eagle_checkpoints
    args = [
        "--served-model-name",
        "m",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "128",
        "--max-num-seqs",
        "4",
        "--pipeline-parallel-size",
        str(pp_size),
        "--enforce-eager",
        "--weight-transfer-config",
        '{"backend": "ipc"}',
        "--speculative-config",
        json.dumps(
            {"method": "eagle", "model": str(draft_path), "num_speculative_tokens": 2}
        ),
    ]
    env = {"VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}
    with RemoteOpenAIServer(str(target_path), args, env_dict=env) as running:
        url = running.url_root
        prompt = "token4 token5 token6"
        original = gen(url, prompt)
        assert ok(original)
        requests.post(f"{url}/pause?mode=abort", timeout=30).raise_for_status()

        baseline = checksums(url, "both")
        target_keys = {key for key in baseline if key.startswith("target:")}
        shared = {
            f"draft:dp0:pp{pp_size - 1}:pcp0:tp0:{name}"
            for name in ("model.embed_tokens.weight", "lm_head.weight")
        }
        assert shared < {key for key in baseline if key.startswith("draft:")}
        assert all(
            baseline[key] == baseline[key.replace("draft:", "target:", 1)]
            for key in shared
        )
        assert checksums(url) == {
            key.removeprefix("target:"): baseline[key] for key in target_keys
        }

        yield url, baseline, shared

        requests.post(f"{url}/resume", timeout=30).raise_for_status()
        recovered = gen(url, prompt)
        assert ok(recovered)
        assert recovered["choices"][0]["text"] == original["choices"][0]["text"]


def assert_eagle_mismatches(url, baseline, expected):
    comparison = weight_checker(url, "compare", baseline, model="both")
    comparison.raise_for_status()
    assert comparison.json() == {"match": not expected, "mismatches": sorted(expected)}


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="IPC weight transfer uses CUDA IPC."
)
def test_eagle_shared_weights_reset_and_recover(eagle_checkpoints):
    _, _, target, draft = eagle_checkpoints
    with paused_eagle_server(eagle_checkpoints, pp_size=1) as (url, baseline, shared):
        target_client = HTTPVLLMWeightSyncClient(url)
        draft_client = DraftWeightSyncClient(url)
        target_keys = {key for key in baseline if key.startswith("target:")}
        draft_keys = baseline.keys() - target_keys
        target_shared = {key.replace("draft:", "target:", 1) for key in shared}

        weight_checker(url, "reset", model="target").raise_for_status()
        assert_eagle_mismatches(url, baseline, target_keys | shared)
        send_ipc_weights(target_client, target)
        assert_eagle_mismatches(url, baseline, set())

        weight_checker(url, "reset", model="draft").raise_for_status()
        assert_eagle_mismatches(url, baseline, draft_keys | target_shared)
        send_ipc_weights(draft_client, draft)
        assert_eagle_mismatches(url, baseline, shared | target_shared)
        draft_with_shared = nn.ModuleDict(
            {
                **dict(draft.named_children()),
                "embed_tokens": target.model.embed_tokens,
                "lm_head": target.lm_head,
            }
        )
        send_ipc_weights(draft_client, draft_with_shared)
        assert_eagle_mismatches(url, baseline, set())

        weight_checker(url, "reset", model="both").raise_for_status()
        assert_eagle_mismatches(url, baseline, baseline.keys())
        cleared = checksums(url, "both")
        weight_checker(url, "reset", model="both").raise_for_status()
        assert checksums(url, "both") == cleared
        send_ipc_weights(target_client, target)
        send_ipc_weights(draft_client, draft)
        assert_eagle_mismatches(url, baseline, set())


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="IPC weight transfer uses CUDA IPC."
)
@multi_gpu_test(num_gpus=2)
def test_eagle_target_reset_and_reload_with_pp(eagle_checkpoints):
    with paused_eagle_server(eagle_checkpoints, pp_size=2) as (url, baseline, shared):
        target_keys = {key for key in baseline if key.startswith("target:")}
        weight_checker(url, "reset", model="target").raise_for_status()
        assert_eagle_mismatches(url, baseline, target_keys | shared)
        requests.post(
            f"{url}/collective_rpc", json={"method": "reload_weights"}, timeout=300
        ).raise_for_status()
        assert_eagle_mismatches(url, baseline, set())

        for model in ("draft", "both"):
            response = weight_checker(url, "reset", model=model)
            assert response.status_code == 409, response.text
            assert "pipeline parallelism" in response.text
            assert_eagle_mismatches(url, baseline, set())


@multi_gpu_test(num_gpus=2)
def test_draft_checksums_use_last_pp_stage_and_reset_is_rejected():
    args = [*draft_args(), "--pipeline-parallel-size", "2"]
    with server(extra_args=args, port=8771, timeout=600) as url:
        requests.post(f"{url}/pause?mode=abort", timeout=30).raise_for_status()
        baseline = checksums(url, "both")
        assert ranks(baseline) == {
            "target:dp0:pp0:pcp0:tp0",
            "target:dp0:pp1:pcp0:tp0",
            "draft:dp0:pp1:pcp0:tp0",
        }
        for model in ("draft", "both"):
            response = weight_checker(url, "reset", model=model)
            assert response.status_code == 409, response.text
            assert (
                "Cannot reset draft weights with pipeline parallelism" in response.text
            )
        comparison = weight_checker(url, "compare", baseline, model="both")
        comparison.raise_for_status()
        assert comparison.json() == {"match": True, "mismatches": []}
