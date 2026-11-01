# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU numerical checks with a local tiny model; no downloaded model is needed."""

import pytest
import torch

from tests.utils import create_new_process_for_each_test


@pytest.mark.parametrize(
    "dtype,eager,layout",
    [
        ("bfloat16", True, "auxiliary_plus_final"),
        ("bfloat16", False, "auxiliary_plus_final"),
        ("float16", True, "auxiliary"),
        ("float16", False, "final"),
    ],
)
@create_new_process_for_each_test()
def test_window_rollout_matches_teacher_forcing(
    tmp_path, monkeypatch, dtype, eager, layout
):
    from transformers import LlamaConfig, LlamaForCausalLM

    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector as files,
    )

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model_path = tmp_path / "model"
    LlamaForCausalLM(config).save_pretrained(model_path)
    llm = LLM(
        model=str(model_path),
        skip_tokenizer_init=True,
        dtype=dtype,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=eager,
        gpu_memory_utilization=0.4,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [0, 1],
                    "hidden_states_layout": layout,
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(tmp_path / "features"),
                "enable_window_capture": True,
                "max_inflight_export_bytes": 1024,
            },
        },
    )
    prompts = [
        {"prompt_token_ids": [1] + [3 + i % 40 for i in range(32)]},
        {"prompt_token_ids": [1, 9, 8, 7]},
        {"prompt_token_ids": [1, 6, 5]},
    ]
    windows = [(0, 8), (2, 8192), (20, 22)]
    params = [
        SamplingParams(
            max_tokens=8,
            ignore_eos=True,
            temperature=0,
            detokenize=False,
            extra_args={
                "kv_transfer_params": {"hidden_state_window": {"start": lo, "end": hi}}
            },
        )
        for lo, hi in windows
    ]
    outputs = llm.generate(prompts, params)
    reference = (
        LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    for output, (lo, hi) in zip(outputs, windows):
        generated = list(output.outputs[0].token_ids)
        if lo >= len(generated):
            assert (
                output.kv_transfer_params["hidden_state_capture_skip_reason"]
                == "insufficient_rows"
            )
            continue
        path = output.kv_transfer_params["hidden_states_path"]
        payload = files.load_hidden_states(path)
        all_ids = list(output.prompt_token_ids) + generated
        positions = torch.arange(
            len(output.prompt_token_ids) - 1 + lo,
            len(output.prompt_token_ids) - 1 + min(hi, len(generated)),
        )
        with torch.no_grad():
            hs = reference(
                torch.tensor([all_ids[:-1]], device="cuda"), output_hidden_states=True
            ).hidden_states
        parts = [hs[0][0], hs[1][0]] if layout != "final" else []
        if layout != "auxiliary":
            parts.append(hs[-1][0])
        expected = torch.stack(parts, dim=1)[positions.to("cuda")].cpu()
        torch.testing.assert_close(
            payload["hidden_states"], expected, atol=0.02, rtol=0.02
        )
        assert payload["hidden_states"].dtype == getattr(torch, dtype)
        assert torch.equal(payload["hidden_positions"], positions)
        assert (
            payload["target_token_ids"].tolist()
            == generated[lo : min(hi, len(generated))]
        )
        assert payload["token_ids"].tolist() == [all_ids[p] for p in positions.tolist()]
        files.cleanup_hidden_states(path)

    near_limit = llm.generate(
        [prompts[0]],
        SamplingParams(
            max_tokens=95,
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            extra_args={
                "kv_transfer_params": {"hidden_state_window": {"start": 88, "end": 96}}
            },
        ),
    )[0]
    limit_path = near_limit.kv_transfer_params["hidden_states_path"]
    limit_payload = files.load_hidden_states(limit_path)
    ids = list(near_limit.prompt_token_ids) + list(near_limit.outputs[0].token_ids)
    with torch.no_grad():
        hs = reference(
            torch.tensor([ids[:-1]], device="cuda"), output_hidden_states=True
        ).hidden_states
    parts = [hs[0][0], hs[1][0]] if layout != "final" else []
    if layout != "auxiliary":
        parts.append(hs[-1][0])
    positions = limit_payload["hidden_positions"]
    assert positions.tolist() == list(range(120, 127))
    expected = torch.stack(parts, dim=1)[positions.to("cuda")].cpu()
    torch.testing.assert_close(
        limit_payload["hidden_states"], expected, atol=0.02, rtol=0.02
    )
    files.cleanup_hidden_states(limit_path)

    eos = llm.generate(
        [prompts[0]],
        SamplingParams(
            max_tokens=8,
            temperature=0,
            detokenize=False,
            logit_bias={2: 100},
            extra_args={
                "kv_transfer_params": {"hidden_state_window": {"start": 0, "end": 8}}
            },
        ),
    )[0]
    assert list(eos.outputs[0].token_ids) == [2]
    eos_path = eos.kv_transfer_params["hidden_states_path"]
    eos_payload = files.load_hidden_states(eos_path)
    assert eos_payload["target_token_ids"].tolist() == [2]
    assert eos_payload["hidden_positions"].tolist() == [len(eos.prompt_token_ids) - 1]
    files.cleanup_hidden_states(eos_path)

    # A real stop-token completion can occur inside the window.
    first = outputs[0].outputs[0].token_ids[0]
    stopped = llm.generate(
        [prompts[0]],
        SamplingParams(
            max_tokens=8,
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            stop_token_ids=[first],
            extra_args={
                "kv_transfer_params": {"hidden_state_window": {"start": 0, "end": 8}}
            },
        ),
    )[0]
    payload = files.load_hidden_states(stopped.kv_transfer_params["hidden_states_path"])
    assert payload["target_token_ids"].tolist() == [first]
    assert payload["hidden_positions"].tolist() == [len(stopped.prompt_token_ids) - 1]
    files.cleanup_hidden_states(stopped.kv_transfer_params["hidden_states_path"])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_graph_replay_masks_nonwindow_and_padding_without_touching_cache(dtype):
    from vllm.model_executor.layers.hidden_state_cache import store_hidden_state_window

    # Offset view and a non-power-of-two width exercise stride and tail masks.
    hidden = torch.arange(9 * 2 * 513, device="cuda").to(dtype).reshape(9, 2, 513)[1:]
    cache = torch.full((4, 4, 2, 513), -3, dtype=dtype, device="cuda")
    slots = torch.tensor([4, 5, 6, 7, 8, 9, 10, -1], device="cuda")
    ranges = torch.tensor([[0, 0], [1, 3], [0, 1], [0, 0]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            store_hidden_state_window(hidden, cache, slots, ranges)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        store_hidden_state_window(hidden, cache, slots, ranges)
    for bounds in ([[0, 0], [1, 3], [0, 1], [0, 0]], [[0, 0], [0, 1], [1, 3], [0, 0]]):
        cache.fill_(-3)
        ranges.copy_(torch.tensor(bounds, device="cuda"))
        graph.replay()
        expected = torch.full_like(cache, -3)
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0 and bounds[slot // 4][0] <= slot % 4 < bounds[slot // 4][1]:
                expected[slot // 4, slot % 4] = hidden[row]
        torch.testing.assert_close(cache, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [128, 2048, 8192])
def test_async_export_and_large_window_staging_preserve_values(tmp_path, dtype, rows):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
        ExampleHiddenStatesConnector,
        PendingSave,
        load_hidden_states,
    )

    connector = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    connector._is_tp_rank_zero = True
    connector._block_size = 16
    connector._kv_cache = (
        torch.arange((rows + 16) * 34, device="cuda")
        .reshape((rows + 16) // 16, 16, 2, 17)
        .remainder(137)
        .to(dtype)
    )
    expected = connector._kv_cache[1:].flatten(0, 1).cpu()
    connector._worker_filenames = {}
    connector._export_budget = 65536
    connector._export_queue = deque()
    connector._copy_stream = None
    connector._layout = "auxiliary"
    connector._capture_metrics = dict(
        copied_bytes=0,
        pinned_live_bytes_upper_bound_peak=0,
        export_submit_seconds=0.0,
        copy_wait_seconds=0.0,
        serialization_seconds=0.0,
    )
    connector._executor = ThreadPoolExecutor(max_workers=2)
    connector._req_futures = {}
    connector._req_copy_events = {}
    connector._lock_fds = {}
    connector.use_lock = True
    for index in range(2):
        path = str(tmp_path / f"{index}.safetensors")
        pending = PendingSave(
            str(index),
            path,
            torch.arange(rows),
            list(range(1, 1 + rows // 16)),
            torch.arange(rows),
            torch.arange(rows),
            torch.tensor([0, 1]),
            torch.tensor([0, rows]),
            torch.arange(rows),
        )
        connector._submit_async_write(pending)
    for event in connector._req_copy_events.values():
        event.synchronize()
    # Simulate allocator reuse as soon as D2H completes, before reading files.
    connector._kv_cache.fill_(-1)
    connector.shutdown()
    for index in range(2):
        payload = load_hidden_states(str(tmp_path / f"{index}.safetensors"))
        torch.testing.assert_close(payload["hidden_states"], expected, atol=0, rtol=0)
    assert connector._capture_metrics["copied_bytes"] == 2 * rows * 34 * 2
    assert connector._capture_metrics["pinned_live_bytes_upper_bound_peak"] <= 65536
