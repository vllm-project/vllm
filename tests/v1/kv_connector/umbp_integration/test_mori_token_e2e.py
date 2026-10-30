# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end restore through the MORI-backed UMBP runtime."""

import time

import pytest

from tests.models.utils import check_logprobs_close
from tests.v1.kv_connector.rocm_pd_accuracy_utils import _REVISIONS, QWEN_MODEL
from tests.v1.kv_connector.umbp_test_utils import mori_has_umbp
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_has_umbp()),
    reason="requires ROCm and MORI built with UMBP",
)


def _external_tokens(llm) -> int:
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if metric.name == "vllm:prompt_tokens_by_source"
        and metric.labels.get("source") == "external_kv_transfer"
    )


def _reset_gpu_prefix_cache(llm) -> None:
    """Drop the GPU prefix cache once UMBP has released the stored blocks."""
    deadline = time.monotonic() + 60
    while not llm.reset_prefix_cache():
        assert time.monotonic() < deadline, "UMBP stores never released GPU blocks"
        llm.generate(
            TokensPrompt(prompt_token_ids=[0]),
            SamplingParams(max_tokens=1),
            use_tqdm=False,
        )


def test_restore_from_umbp_matches_recompute(vllm_runner, socket_dir):
    prompts = ["The first ten natural numbers are " * 32]
    with vllm_runner(
        QWEN_MODEL,
        revision=_REVISIONS[QWEN_MODEL],
        enforce_eager=True,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UMBPStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "mode": "embedded",
                "capacity_bytes": 1 << 30,
                "lookup_dir": socket_dir,
            },
        ),
    ) as runner:
        computed = runner.generate_greedy_logprobs(prompts, 16, 5)
        _reset_gpu_prefix_cache(runner.llm)
        before = _external_tokens(runner.llm)
        restored = runner.generate_greedy_logprobs(prompts, 16, 5)
        assert _external_tokens(runner.llm) > before

    check_logprobs_close(
        outputs_0_lst=computed,
        outputs_1_lst=restored,
        name_0="computed",
        name_1="restored",
    )
