# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch

from vllm.platforms import CpuArchEnum, current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.cpu.logprob import (
    CPUPromptLogprobsWorker,
    LogprobTokenIdsState,
    compute_prompt_logprobs_with_chunking,
    compute_token_logprobs,
    compute_topk_scores,
    get_prompt_logprobs_token_ids,
)

if current_platform.is_cpu():
    import vllm._C  # noqa: F401

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


@pytest.mark.parametrize("logits_mode", [False, True])
def test_cpu_output_logprobs(logits_mode: bool) -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    sampled_token_ids = torch.tensor([2, 1], dtype=torch.int64)

    output = compute_topk_scores(
        logits,
        num_logprobs=2,
        sampled_token_ids=sampled_token_ids,
        cu_num_logits=[0, 1, 2],
        max_per_req_token_ids=0,
        logits_mode=logits_mode,
    )

    assert output.logprob_token_ids.tolist() == [
        [2, 0, 1],
        [1, 3, 1],
    ]
    assert output.logprob_token_ids.dtype == torch.int64
    expected_scores = torch.tensor([[1.0, 4.0, 2.0], [2.0, 3.0, 2.0]])
    if not logits_mode:
        expected_scores -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    torch.testing.assert_close(output.logprobs, expected_scores)
    assert output.selected_token_ranks.tolist() == [3, 2]
    assert output.cu_num_generated_tokens == [0, 1, 2]


def test_cpu_prompt_logprobs_scores_prompt_targets() -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )

    prompt_token_ids = get_prompt_logprobs_token_ids(
        num_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        num_computed_tokens=torch.tensor([0], dtype=torch.int32),
        all_token_ids=torch.tensor([[7, 1, 2]], dtype=torch.int32),
    )
    token_ids, logprobs, token_ranks = compute_prompt_logprobs_with_chunking(
        prompt_token_ids=prompt_token_ids,
        prompt_hidden_states=logits,
        logits_fn=lambda hidden_states: hidden_states,
        num_prompt_logprobs=2,
    )

    assert token_ids.tolist() == [[1, 0, 1], [2, 3, 1]]
    expected = torch.tensor([[2.0, 4.0, 2.0], [1.0, 3.0, 2.0]])
    expected -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    torch.testing.assert_close(logprobs, expected)
    assert token_ranks.tolist() == [2, 3]


def _input_batch(
    req_id: str,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    prefill_len: int,
) -> InputBatch:
    buffers = InputBuffers(1, num_scheduled_tokens, torch.device("cpu"))
    batch = InputBatch.make_dummy(1, num_scheduled_tokens, buffers)
    batch.req_ids = [req_id]
    batch.num_computed_prefill_tokens_np[:] = num_computed_tokens
    batch.num_scheduled_tokens[:] = num_scheduled_tokens
    batch.prefill_len_np[:] = prefill_len
    return batch


def test_cpu_prompt_logprobs_accumulate_chunked_prefill() -> None:
    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    worker.add_request("request", 0, SamplingParams(prompt_logprobs=1))

    logits = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 3.0],
            [1.0, 4.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
        ]
    )
    all_token_ids = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.int32)
    prompt_lens = np.array([4], dtype=np.int32)

    first = worker.compute_prompt_logprobs(
        lambda hidden_states: hidden_states,
        logits[:2],
        _input_batch("request", 0, 2, 4),
        all_token_ids,
        torch.tensor([0], dtype=torch.int32),
        prompt_lens,
    )
    assert first == {}

    second = worker.compute_prompt_logprobs(
        lambda hidden_states: hidden_states,
        logits[2:],
        _input_batch("request", 2, 2, 4),
        all_token_ids,
        torch.tensor([2], dtype=torch.int32),
        prompt_lens,
    )

    result = second["request"]
    assert result.logprob_token_ids[:, 0].tolist() == [1, 2, 3]
    expected = torch.log_softmax(logits[:3], dim=-1)[
        torch.arange(3), torch.tensor([1, 2, 3])
    ]
    torch.testing.assert_close(result.logprobs[:, 0], expected)


def test_cpu_token_logprobs_accept_int32_token_ids() -> None:
    logits = torch.tensor(
        [[4.0, 2.0, 1.0, 0.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    token_ids = torch.tensor([[1, 3], [2, 0]], dtype=torch.int32)

    actual = compute_token_logprobs(logits, token_ids)
    expected = logits.gather(-1, token_ids).float()
    expected -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected)


def test_cpu_prompt_logprobs_honor_start_position() -> None:
    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    worker.add_request(
        "request",
        0,
        SamplingParams(prompt_logprobs=1, prompt_logprob_start=2),
    )
    logits = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 3.0],
            [1.0, 4.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
        ]
    )
    all_token_ids = torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.int32)

    result = worker.compute_prompt_logprobs(
        lambda hidden_states: hidden_states,
        logits,
        _input_batch("request", 0, 4, 4),
        all_token_ids,
        torch.tensor([0], dtype=torch.int32),
        np.array([4], dtype=np.int32),
    )["request"]

    assert result.logprob_token_ids[:, 0].tolist() == [3]


def test_cpu_requested_prompt_token_scores_honor_start_position() -> None:
    worker = CPUPromptLogprobsWorker(max_num_reqs=1, device=torch.device("cpu"))
    token_ids = np.array([[0, 2], [1, 3]], dtype=np.int64)
    worker.add_request(
        "request",
        0,
        SamplingParams(
            prompt_logprob_token_ids=token_ids,
            prompt_logprob_start=1,
        ),
    )
    logits = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 3.0],
            [1.0, 4.0, 3.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
        ]
    )

    result = worker.compute_prompt_token_id_logprobs(
        lambda hidden_states: hidden_states,
        logits,
        _input_batch("request", 0, 4, 4),
        np.array([4], dtype=np.int32),
    )["request"]
    expected = compute_token_logprobs(logits[1:3], torch.from_numpy(token_ids))

    torch.testing.assert_close(result, expected)


def _make_logprob_token_ids_state(
    num_token_ids: list[int], token_ids: list[list[int]]
) -> LogprobTokenIdsState:
    state = LogprobTokenIdsState(len(num_token_ids), torch.device("cpu"))
    for req_idx, num_ids in enumerate(num_token_ids):
        state.add_request(
            req_idx,
            SamplingParams(logprob_token_ids=token_ids[req_idx][:num_ids]),
        )
    return state


@pytest.mark.parametrize("logits_mode", [False, True])
def test_cpu_requested_output_logprobs(logits_mode: bool) -> None:
    logits = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 2.0, 1.0, 5.0, 4.0, 3.0],
            [1.0, 6.0, 2.0, 5.0, 3.0, 4.0],
            [4.0, 1.0, 6.0, 0.0, 5.0, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    sampled_token_ids = torch.tensor([4, 1, 0, 5], dtype=torch.int64)
    state = _make_logprob_token_ids_state(
        num_token_ids=[0, 5, 2, 3],
        token_ids=[
            [0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4],
            [5, 2, 0, 0, 0],
            [4, 1, 3, 0, 0],
        ],
    )

    output = compute_topk_scores(
        logits,
        num_logprobs=4,
        sampled_token_ids=sampled_token_ids,
        cu_num_logits=[0, 1, 2, 3, 4],
        num_per_req_token_ids=state.num_token_ids,
        per_req_token_ids=state.token_ids,
        expanded_idx_mapping=torch.tensor([2, 0, 3, 1], dtype=torch.int64),
        max_per_req_token_ids=5,
        logits_mode=logits_mode,
    )

    assert output.logprob_token_ids.tolist() == [
        [4, 5, 2, 0, 0, 0],
        [1, 3, 4, 5, 1, 0],
        [0, 4, 1, 3, 0, 0],
        [5, 0, 1, 2, 3, 4],
    ]
    expected_scores = torch.tensor(
        [
            [2.0, 1.0, 4.0, -math.inf, -math.inf, -math.inf],
            [2.0, 5.0, 4.0, 3.0, 2.0, -math.inf],
            [1.0, 3.0, 6.0, 5.0, -math.inf, -math.inf],
            [2.0, 4.0, 1.0, 6.0, 0.0, 5.0],
        ]
    )
    if not logits_mode:
        expected_scores -= torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    torch.testing.assert_close(output.logprobs, expected_scores)
    assert output.selected_token_ranks.tolist() == [5, 4, 6, 4]
    assert output.cu_num_generated_tokens == [0, 1, 2, 3, 4]


def test_cpu_requested_output_logprobs_without_topk() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    state = _make_logprob_token_ids_state(
        num_token_ids=[0, 2],
        token_ids=[[0, 0], [3, 1]],
    )

    output = compute_topk_scores(
        logits,
        num_logprobs=0,
        sampled_token_ids=torch.tensor([2, 1]),
        cu_num_logits=None,
        num_per_req_token_ids=state.num_token_ids,
        per_req_token_ids=state.token_ids,
        expanded_idx_mapping=torch.tensor([1, 0], dtype=torch.int64),
        max_per_req_token_ids=2,
        logits_mode=True,
    )

    assert output.logprob_token_ids.tolist() == [[2, 3, 1], [1, 0, 0]]
    torch.testing.assert_close(
        output.logprobs,
        torch.tensor([[2.0, 1.0, 3.0], [2.0, -math.inf, -math.inf]]),
    )
    assert output.selected_token_ranks.tolist() == [3, 3]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("offset", [0.0, 16384.0, -16384.0, 32768.0, -32768.0])
def test_cpu_token_logprobs_preserve_common_offsets(dtype, offset):
    # Use the actual rounded input values as the reference, including FP16 ties.
    storage = torch.tensor(
        [
            [-4.0, 99.0, 0.0, 99.0, 1.0, 99.0, 3.0, 99.0],
            [2.0, 99.0, -1.0, 99.0, 4.0, 99.0, 0.0, 99.0],
        ],
        dtype=dtype,
    )
    logits = (storage + offset)[:, ::2]
    original = logits.clone()
    token_ids = torch.tensor([[3, 1, 1], [2, 0, 3]], dtype=torch.int32)
    expected = logits.double().log_softmax(-1).gather(-1, token_ids.long()).float()

    actual = compute_token_logprobs(logits, token_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("offset", [1e8, -1e8])
def test_cpu_token_logprobs_uniform_large_offset(offset):
    logits = torch.full((3, 16), offset)
    ids = torch.tensor([[0, 15], [9, 9], [4, 7]])
    expected = torch.full((3, 2), -math.log(16))
    torch.testing.assert_close(
        compute_token_logprobs(logits, ids), expected, rtol=1e-6, atol=1e-6
    )


def test_cpu_token_logprobs_preserve_masked_and_nonfinite_inputs():
    logits = torch.tensor(
        [
            [-math.inf, 1.0, 2.0, 3.0],
            [-math.inf, -math.inf, -math.inf, -math.inf],
            [1.0, math.inf, 2.0, -math.inf],
            [1.0, math.nan, 2.0, 3.0],
        ]
    )
    ids = torch.tensor([[0, 2], [1, 3], [0, 1], [2, 3]])
    # Preserve the existing semantics of invalid all-masked/nonfinite rows.
    expected = logits.gather(-1, ids) - torch.logsumexp(logits, -1, keepdim=True)
    actual = compute_token_logprobs(logits, ids)
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("shape", [(0, 16), (3, 16)])
def test_cpu_token_logprobs_empty_selected_tokens(shape):
    logits = torch.ones(shape)
    ids = torch.empty((shape[0], 0), dtype=torch.int32)
    actual = compute_token_logprobs(logits, ids)
    assert actual.shape == ids.shape
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("logits_mode", [False, True])
def test_cpu_topk_and_prompt_scores_with_common_offset(logits_mode):
    logits = torch.full((3, 16), 1e8)
    ids = torch.tensor([2, 5, 9])
    result = compute_topk_scores(logits, 3, ids, logits_mode=logits_mode)
    expected_value = 1e8 if logits_mode else -math.log(16)
    torch.testing.assert_close(
        result.logprobs,
        torch.full_like(result.logprobs, expected_value),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.logprob_token_ids[:, 0].tolist() == ids.tolist()
    assert result.selected_token_ranks.tolist() == [16, 16, 16]
    _, scores, _ = compute_prompt_logprobs_with_chunking(
        ids,
        logits,
        lambda hidden: hidden,
        3,
        logprobs_mode="raw_logits" if logits_mode else "raw_logprobs",
    )
    torch.testing.assert_close(
        scores, torch.full_like(scores, expected_value), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("offset", [0.0, 1e8, -1e8])
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
def test_standalone_cpu_engine_logprobs_with_uniform_large_logits(
    tmp_path, vllm_runner, offset, logprobs_mode
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, TokenizersBackend

    # A real model whose final layernorm and output projection produce known
    # uniform logits. Both prompt scoring and autoregressive decoding still run.
    config = GPT2Config(
        vocab_size=16,
        n_positions=32,
        n_embd=32,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(37)
        reference = GPT2LMHeadModel(config).eval()
        reference.transformer.ln_f.weight.zero_()
        reference.transformer.ln_f.bias.fill_(1.0)
        reference.lm_head.weight.zero_()
        reference.lm_head.weight[:, 0].fill_(offset)
        reference.save_pretrained(tmp_path)
    vocabulary = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
    }
    vocabulary.update({f"unused{i}": i for i in range(7, 16)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        dtype="float32",
        enforce_eager=True,
        max_model_len=32,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        enable_prefix_caching=False,
        logprobs_mode=logprobs_mode,
    ) as model:
        outputs = model.llm.generate(
            ["alpha beta gamma", "gamma alpha"],
            SamplingParams(
                temperature=0,
                max_tokens=2,
                ignore_eos=True,
                logprobs=-1,
                prompt_logprobs=-1,
            ),
            use_tqdm=False,
        )
    for result in outputs:
        assert result.prompt_logprobs[0] is None
        rows = [*result.prompt_logprobs[1:], *result.outputs[0].logprobs]
        assert len(result.outputs[0].token_ids) == 2
        for row in rows:
            assert set(row) == set(range(16))
            assert [item.logprob for item in row.values()] == pytest.approx(
                [-math.log(16)] * 16, rel=1e-6, abs=1e-6
            )
