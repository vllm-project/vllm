# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import SamplingParams

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


def test_tensor_dump_records_real_model_forward(
    tmp_path, monkeypatch: pytest.MonkeyPatch, vllm_runner
):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        MODEL,
        enforce_eager=True,
        max_model_len=32,
        max_num_seqs=1,
        debug_tensor_dump_output_folder=str(tmp_path),
        debug_tensor_dump_layers=[0],
    ) as runner:
        outputs = runner.llm.generate(
            ["Hello"], SamplingParams(temperature=0, max_tokens=1)
        )
    assert len(outputs[0].outputs[0].token_ids) == 1

    dump_files = sorted(tmp_path.glob("rank*/Pass*.pt"))
    assert dump_files
    record = torch.load(dump_files[0], weights_only=False)

    input_ids = record["vllm.forward_batch_info.input_ids"]
    positions = record["vllm.forward_batch_info.positions"]
    seq_lens = record["vllm.forward_batch_info.extend_seq_lens"]
    num_tokens = int(seq_lens.sum())
    assert input_ids.numel() == positions.numel() == num_tokens
    assert len(record["vllm.forward_batch_info.rids"]) == len(seq_lens)

    layer_name = next(name for name in record if name.endswith("layers.0"))
    hidden_states, residual = record[layer_name]
    assert hidden_states.shape[0] == residual.shape[0] == num_tokens
