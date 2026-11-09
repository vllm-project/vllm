# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check tensor-parallel cache isolation for head-sharded direct KV pages."""

import time

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_marks
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig

MODEL = "Qwen/Qwen3-0.6B"
PROMPT_TOKEN_IDS = list(range(1000, 2000))
BLOCK_SIZE = 16

pytestmark = multi_gpu_marks(num_gpus=2)


def _num_stored_files(root_dir):
    return sum(1 for _ in root_dir.rglob("*.bin"))


def _generate(vllm_runner, tensor_parallel_size, root_dir):
    stored_before = _num_stored_files(root_dir)
    with vllm_runner(
        MODEL,
        tensor_parallel_size=tensor_parallel_size,
        block_size=BLOCK_SIZE,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.3,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 2_000_000_000,
                "secondary_tiers": [{"type": "fs", "root_dir": str(root_dir)}],
            },
        ),
    ) as runner:
        tokenizer = runner.llm.get_tokenizer()
        (output,) = runner.llm.generate(
            TokensPrompt(prompt_token_ids=PROMPT_TOKEN_IDS),
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True, logprobs=5),
            use_tqdm=False,
        )
        if output.num_cached_tokens == 0:
            # Engine shutdown cancels queued stores.
            expected = stored_before + len(PROMPT_TOKEN_IDS) // BLOCK_SIZE
            deadline = time.monotonic() + 60
            while _num_stored_files(root_dir) < expected:
                assert time.monotonic() < deadline, "KV blocks were not stored"
                time.sleep(0.1)
    completion = output.outputs[0]
    return output.num_cached_tokens, (
        list(completion.token_ids),
        tokenizer.decode(completion.token_ids),
        completion.logprobs,
    )


def test_direct_layout_is_not_shared_across_tp_sizes(vllm_runner, tmp_path):
    cached_by_writer, _ = _generate(vllm_runner, 1, tmp_path)
    assert cached_by_writer == 0

    cached_cold, cold = _generate(vllm_runner, 2, tmp_path)
    assert cached_cold == 0

    cached_reuse, reuse = _generate(vllm_runner, 2, tmp_path)
    assert cached_reuse > 0
    check_logprobs_close(
        outputs_0_lst=[cold],
        outputs_1_lst=[reuse],
        name_0="computed at TP=2",
        name_1="restored at TP=2",
    )
