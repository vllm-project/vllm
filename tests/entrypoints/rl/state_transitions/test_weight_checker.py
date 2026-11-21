# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from unittest.mock import patch

import pytest
import regex as re
import requests
import torch
from transformers import AutoModelForCausalLM

from tests.entrypoints.rl.conftest import (
    MODEL_NAME,
    gen,
    health,
    ok,
    server,
)
from tests.utils import multi_gpu_test
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
            {"action": "reset", "model": None},
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

    @pytest.mark.parametrize("model", ["draft", "both"])
    def test_missing_draft_rejects_all_actions_without_resetting_target(
        self, server_url, model
    ):
        target = checksums(server_url)
        for action in ("checksum", "reset", "compare"):
            response = weight_checker(server_url, action, baseline={}, model=model)
            assert response.status_code == 409, response.text
            assert "resident speculative draft model" in response.text
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
        WeightTransferTrainerFactory.trainer_init(
            IPCTrainerInitInfo(rank=0),
            client=HTTPVLLMWeightSyncClient(url),
            source=ModuleSource(trainer_model),
        ).send_weights()
        assert weight_checker(url, "compare", baseline).json() == {
            "match": True,
            "mismatches": [],
        }


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "args, expected",
    [
        (["--data-parallel-size", "2"], {"dp0:pp0:pcp0:tp0", "dp1:pp0:pcp0:tp0"}),
        (["--tensor-parallel-size", "2"], {"dp0:pp0:pcp0:tp0", "dp0:pp0:pcp0:tp1"}),
    ],
)
def test_checksum_and_reset_cover_every_worker(args, expected):
    with server(extra_args=args, port=8771, timeout=600) as url:
        assert ranks(checksums(url)) == expected
        reset_reload_and_compare(url)


DRAFT_MODEL = "Qwen/Qwen3-0.6B"


def draft_args():
    return [
        "--speculative-config",
        json.dumps(
            {
                "method": "draft_model",
                "model": DRAFT_MODEL,
                "num_speculative_tokens": 2,
            }
        ),
    ]


class DraftWeightSyncClient(HTTPVLLMWeightSyncClient):
    def start_weight_update(self) -> None:
        self._post("start_draft_weight_update")
        response = weight_checker(self.base_url, "reset", model="both")
        assert response.status_code == 409, response.text
        assert "active or incomplete weight update" in response.text


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
        both.update({f"draft:{key}": digest for key, digest in draft.items()})
        assert target and draft
        assert checksums(url) == target
        assert checksums(url, "both") == both

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
                is_draft = restored_model == "draft"
                trainer_model = AutoModelForCausalLM.from_pretrained(
                    DRAFT_MODEL if is_draft else MODEL_NAME, torch_dtype=torch.bfloat16
                ).cuda()
                client = (
                    DraftWeightSyncClient(url)
                    if is_draft
                    else HTTPVLLMWeightSyncClient(url)
                )
                WeightTransferTrainerFactory.trainer_init(
                    IPCTrainerInitInfo(rank=0),
                    client=client,
                    source=ModuleSource(trainer_model),
                ).send_weights()
                del trainer_model

            comparison = weight_checker(url, "compare", baseline, model=model)
            comparison.raise_for_status()
            assert comparison.json() == {"match": True, "mismatches": []}
        requests.post(f"{url}/resume", timeout=30).raise_for_status()
        assert ok(gen(url))


@multi_gpu_test(num_gpus=4)
def test_both_checksums_cover_every_dp_tp_shard():
    args = [*draft_args(), "--data-parallel-size", "2", "--tensor-parallel-size", "2"]
    with server(extra_args=args, port=8771, timeout=600) as url:
        requests.post(f"{url}/pause?mode=abort", timeout=30).raise_for_status()
        expected = {
            f"{model}:dp{dp}:pp0:pcp0:tp{tp}"
            for model in ("target", "draft")
            for dp in range(2)
            for tp in range(2)
        }
        baseline = checksums(url, "both")
        assert ranks(baseline) == expected
        weight_checker(url, "reset", model="both").raise_for_status()
        comparison = weight_checker(url, "compare", baseline, model="both")
        comparison.raise_for_status()
        assert ranks(comparison.json()["mismatches"]) == expected


@multi_gpu_test(num_gpus=2)
def test_draft_checksum_and_reset_use_only_the_last_pp_stage():
    args = [*draft_args(), "--pipeline-parallel-size", "2"]
    with server(extra_args=args, port=8771, timeout=600) as url:
        requests.post(f"{url}/pause?mode=abort", timeout=30).raise_for_status()
        assert ranks(checksums(url, "both")) == {
            "target:dp0:pp0:pcp0:tp0",
            "target:dp0:pp1:pcp0:tp0",
            "draft:dp0:pp1:pcp0:tp0",
        }
        target = checksums(url, "target")
        draft = checksums(url, "draft")
        weight_checker(url, "reset", model="draft").raise_for_status()
        assert checksums(url, "target") == target
        assert checksums(url, "draft") != draft
