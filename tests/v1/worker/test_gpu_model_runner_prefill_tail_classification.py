# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`GPUModelRunner._prepare_inputs` must tell a chunked-prefill continuation
row apart from a genuine zero-draft decode row.

Both kinds of row are absent from `scheduled_spec_decode_tokens` and both
schedule exactly one token, so `num_decode_draft_tokens` starts at the
placeholder `-1` for either. The `zero_draft_decode_mask` in
`gpu_model_runner.py` is what tells them apart afterwards, using
`num_computed_tokens >= num_prompt_tokens`: a prefill continuation must keep
`-1` (it carries no drafts and no accepted-token offset to apply), while a
zero-draft decode row must become `0` (it still has to run the speculative
path so the recurrent backends apply that offset). Getting this wrong in
either direction was the root cause behind issue #40875: Kimi-Linear crashed
with an illegal memory access when prefill tails were misclassified as
zero-draft decodes.

Each batch below also includes one row with a real draft token, so
`scheduled_spec_decode_tokens` is non-empty even under the pre-fix
`use_spec_decode = len(scheduled_spec_decode_tokens) > 0` predicate. That
keeps the assertions targeted at the mask fix alone, not the unrelated
`use_spec_decode` predicate fix (`self.speculative_config is not None`) that
landed in the same patch.
"""

import numpy as np
import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

BLOCK_SIZE = 16
NUM_BLOCKS = 10
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_vllm_config() -> VllmConfig:
    model_config = ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42)
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto"
    )
    cache_config.kv_cache_layout = "LBNHC"
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=ParallelConfig(),
        # ngram needs no draft model weights, so the fixture stays as cheap as
        # the speculative-decoding-free one in test_gpu_model_runner.py.
        speculative_config=SpeculativeConfig(method="ngram", num_speculative_tokens=3),
    )


def _initialize_kv_cache(runner: GPUModelRunner) -> None:
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=tensor_size,
                layers=["layer.0"],
                layer_stride=tensor_size,
                block_stride=attn_spec.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
        kernel_block_sizes=[
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        ],
        max_num_blocks_per_req=[NUM_BLOCKS],
    )
    runner.initialize_attn_backend(kv_cache_config)


@pytest.fixture
def spec_decode_model_runner(dist_init):
    vllm_config = _get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
            num_heads, head_size, 0.1
        )
        runner = GPUModelRunner(vllm_config, DEVICE_TYPE)
        _initialize_kv_cache(runner)
        yield runner


def _new_request(
    req_id: str, prompt_len: int, num_computed_tokens: int
) -> tuple[NewRequestData, int]:
    """A request with `prompt_len` prompt tokens, `num_computed_tokens` of
    which are already computed (e.g. a prefix-cache hit), scheduled for the
    single remaining/next token this step."""
    return (
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(range(prompt_len)),
            mm_features=[],
            sampling_params=SamplingParams(),
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
        ),
        1,
    )


def test_prefill_continuation_keeps_placeholder_decode_zero_draft_becomes_zero(
    spec_decode_model_runner,
):
    """One request has one token left to prefill (`num_computed_tokens ==
    num_prompt_tokens - 1`); another has already finished its prompt and is
    now decoding with no draft tokens proposed this step; a third is decoding
    with a real draft. `decode_with_draft` gives the batch a non-empty
    `scheduled_spec_decode_tokens`, so this also exercises the `-1`-vs-`0`
    distinction on the pre-fix definition of `use_spec_decode` (`len(sched
    uled_spec_decode_tokens) > 0`), isolating the mask fix from the unrelated
    `use_spec_decode` predicate fix that landed in the same patch."""
    runner = spec_decode_model_runner

    prefill_req, prefill_sched = _new_request(
        "prefill_tail", prompt_len=11, num_computed_tokens=10
    )
    decode_req, decode_sched = _new_request(
        "decode_zero_draft", prompt_len=3, num_computed_tokens=3
    )
    drafted_req, _ = _new_request(
        "decode_with_draft", prompt_len=3, num_computed_tokens=3
    )
    num_scheduled_tokens = {
        "prefill_tail": prefill_sched,
        "decode_zero_draft": decode_sched,
        "decode_with_draft": 2,  # 1 draft token + 1 bonus token
    }
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[prefill_req, decode_req, drafted_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={"decode_with_draft": [99]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    runner._update_states(scheduler_output)
    runner._prepare_inputs(
        scheduler_output,
        np.array(
            [num_scheduled_tokens[r] for r in runner.input_batch.req_ids],
            dtype=np.int32,
        ),
    )

    prefill_idx = runner.input_batch.req_id_to_index["prefill_tail"]
    decode_idx = runner.input_batch.req_id_to_index["decode_zero_draft"]
    drafted_idx = runner.input_batch.req_id_to_index["decode_with_draft"]
    assert runner.num_decode_draft_tokens.np[prefill_idx] == -1
    assert runner.num_decode_draft_tokens.np[decode_idx] == 0
    assert runner.num_decode_draft_tokens.np[drafted_idx] == 1


def test_prefill_tail_of_two_or_three_tokens_also_keeps_the_placeholder(
    spec_decode_model_runner,
):
    """A chunked-prefill tail can be 1, 2, or 3 tokens (`long_prefill_token_
    threshold` rounding); the mask only fires for `num_scheduled_tokens == 1`,
    so wider tails must keep `-1` regardless of how much prompt is left. A
    third, drafted decode row keeps `scheduled_spec_decode_tokens` non-empty
    so the assertions hold on the pre-fix `use_spec_decode` predicate too."""
    runner = spec_decode_model_runner

    tail_2_req, _ = _new_request("tail_2", prompt_len=12, num_computed_tokens=10)
    tail_3_req, _ = _new_request("tail_3", prompt_len=13, num_computed_tokens=10)
    drafted_req, _ = _new_request(
        "decode_with_draft", prompt_len=3, num_computed_tokens=3
    )
    num_scheduled_tokens = {"tail_2": 2, "tail_3": 3, "decode_with_draft": 2}
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[tail_2_req, tail_3_req, drafted_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={"decode_with_draft": [99]},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    runner._update_states(scheduler_output)
    runner._prepare_inputs(
        scheduler_output,
        np.array(
            [num_scheduled_tokens[r] for r in runner.input_batch.req_ids],
            dtype=np.int32,
        ),
    )

    tail_2_idx = runner.input_batch.req_id_to_index["tail_2"]
    tail_3_idx = runner.input_batch.req_id_to_index["tail_3"]
    assert runner.num_decode_draft_tokens.np[tail_2_idx] == -1
    assert runner.num_decode_draft_tokens.np[tail_3_idx] == -1
