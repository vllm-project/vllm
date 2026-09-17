# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2 tiny-MoE numerical E2E coverage, run by Intel Buildkite.

Uses the real engine and model implementation, not a hand-written operator chain.
No checkpoint or tokenizer downloads; all vocabulary logits are compared exactly.
"""

from unittest.mock import patch

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_xpu(), reason="XPU batch-invariance tests"
    ),
    pytest.mark.distributed(num_gpus=2),
]

_VOCAB_SIZE = 1024
_TRACE = list(range(11, 43))


@pytest.fixture(autouse=True)
def enable_batch_invariance(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


@pytest.fixture
def tiny_qwen3_moe_path(tmp_path):
    """Create a local Qwen3-MoE config for dummy-loading tests."""
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=_VOCAB_SIZE,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config.save_pretrained(tmp_path)
    return tmp_path


def _initialize_weights(worker):
    """Avoid tiny dummy norm weights hiding changes in attention or experts."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights

    model = worker.get_model()
    initialize_dummy_weights(model, worker.model_config, low=-0.05, high=0.05)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, RMSNorm):
                layer.weight.fill_(1.0)


def _check_xccl_collectives(worker):
    """Exercise real TP2 collectives even when the model does not use RS."""
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    communicator = group.device_communicator
    assert communicator is not None
    assert group.world_size == 2
    rank = group.rank_in_group
    device = worker.device
    baseline = None
    for num_tokens in (2, 8, 300):
        rows = torch.arange(num_tokens * 1024, device=device, dtype=torch.float32)
        rows = rows.reshape(num_tokens, 1024) / 16
        input_ = rows + rank
        expected = rows * 2 + 1
        reduced = communicator.all_reduce(input_)
        torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
        if baseline is None:
            baseline = reduced[0].clone()
        else:
            torch.testing.assert_close(reduced[0], baseline, rtol=0, atol=0)
        for sizes in (None, [1, num_tokens - 1]):
            if sizes is None:
                actual = communicator.reduce_scatter(input_, dim=0)
                reference = expected.chunk(2)[rank]
            else:
                actual = communicator.reduce_scatterv(input_, dim=0, sizes=sizes)
                reference = expected.split(sizes)[rank]
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def _logits(output):
    completion = output.outputs[0]
    assert completion.token_ids == _TRACE
    assert completion.logprobs is not None
    assert len(completion.logprobs) == len(_TRACE)
    for step in completion.logprobs:
        assert set(step) == set(range(_VOCAB_SIZE))
    values = torch.tensor(
        [
            [step[token_id].logprob for token_id in range(_VOCAB_SIZE)]
            for step in completion.logprobs
        ],
        dtype=torch.float32,
    )
    assert torch.isfinite(values).all()
    assert (values.amax(dim=-1) > values.amin(dim=-1)).all()
    return values


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
@pytest.mark.timeout(600)
@create_new_process_for_each_test("spawn")
def test_tiny_moe_logits_across_batches(
    tiny_qwen3_moe_path, enforce_eager, monkeypatch, vllm_runner
):
    """Keep token history fixed while changing batch position and scheduling."""
    if torch.xpu.device_count() < 2:
        pytest.skip("Requires two XPUs")

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.triton_utils import HAS_TRITON
    from vllm.v1.engine.core_client import InprocClient

    assert HAS_TRITON, "XPU batch invariance requires a working Intel Triton"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    def prompt(length, offset=0):
        return TokensPrompt(
            prompt_token_ids=[3 + (i + offset) % 253 for i in range(length)]
        )

    params = SamplingParams(
        temperature=0,
        max_tokens=len(_TRACE),
        logprobs=-1,
        detokenize=False,
        trace_decode_token_ids=_TRACE,
    )
    needle = prompt(257)
    filler_lengths = (
        7,
        31,
        63,
        95,
        127,
        159,
        191,
        223,
        255,
        287,
        319,
        383,
        511,
        639,
        767,
        895,
    )
    fillers = [prompt(length, i + 1) for i, length in enumerate(filler_lengths)]

    with vllm_runner(
        str(tiny_qwen3_moe_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        seed=0,
        enforce_eager=enforce_eager,
        enable_trace_replay=True,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        max_model_len=1024,
        max_num_seqs=32,
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.15,
        kv_cache_memory_bytes=256 * 1024 * 1024,
    ) as runner:
        llm = runner.llm
        llm.collective_rpc(_initialize_weights)
        llm.collective_rpc(_check_xccl_collectives)
        baseline = _logits(llm.generate([needle], params, use_tqdm=False)[0])
        for position in (0, 8, 16):
            batch = list(fillers)
            batch.insert(position, needle)
            output = llm.generate(batch, params, use_tqdm=False)[position]
            torch.testing.assert_close(
                _logits(output),
                baseline,
                rtol=0,
                atol=0,
                msg=f"Needle logits changed at batch position {position}",
            )

        engine = llm.llm_engine
        assert isinstance(engine.engine_core, InprocClient)
        scheduler = engine.engine_core.engine_core.scheduler
        schedule = scheduler.schedule
        mixed_batches = []
        streaming_params = params.clone()
        streaming_params.output_kind = RequestOutputKind.CUMULATIVE
        needle_id = engine.add_request("needle", needle, streaming_params)

        def observe_schedule():
            is_prefill = {
                request_id: req.num_computed_tokens < req.num_prompt_tokens
                for request_id, req in scheduler.requests.items()
            }
            output = schedule()
            scheduled = output.num_scheduled_tokens
            mixed_batches.append(
                needle_id in scheduled
                and not is_prefill[needle_id]
                and any(is_prefill[req_id] for req_id in scheduled)
            )
            return output

        inserted = False
        final_output = None
        with patch.object(scheduler, "schedule", side_effect=observe_schedule):
            for _ in range(64):
                if not engine.has_unfinished_requests():
                    break
                outputs = engine.step()
                for output in outputs:
                    if output.request_id != "needle":
                        continue
                    if output.finished:
                        final_output = output
                    if (
                        not inserted
                        and not output.finished
                        and output.outputs[0].token_ids
                    ):
                        for i, filler in enumerate(fillers):
                            engine.add_request(f"filler-{i}", filler, params)
                        inserted = True
            assert not engine.has_unfinished_requests(), "Requests did not finish"
        assert inserted and any(mixed_batches), "No mixed prefill/decode was exercised"
        assert final_output is not None
        torch.testing.assert_close(
            _logits(final_output),
            baseline,
            rtol=0,
            atol=0,
            msg="Needle logits changed when new prefills joined decode",
        )
