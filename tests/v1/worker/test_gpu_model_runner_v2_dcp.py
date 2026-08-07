# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu import model_runner as mrv2


def test_dummy_dcp_lengths_are_attached_before_attention_metadata(monkeypatch):
    events = []
    is_sm90 = True
    expect_dcp = True
    monkeypatch.setattr(
        mrv2.current_platform,
        "is_device_capability",
        lambda _: is_sm90,
    )
    dcp_local_seq_lens = torch.zeros(2, dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=1,
        seq_lens=torch.tensor([4], dtype=torch.int32),
        dcp_local_seq_lens=None,
    )

    monkeypatch.setattr(
        mrv2,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            SimpleNamespace(num_reqs=1, num_tokens=1, cg_mode=None),
            None,
        ),
    )
    monkeypatch.setattr(
        mrv2.InputBatch, "make_dummy", lambda *args, **kwargs: input_batch
    )

    def prepare_dcp(output, seq_lens, num_reqs, dcp_size, dcp_rank, interleave):
        events.append("prepare_dcp")
        assert output is dcp_local_seq_lens
        assert seq_lens is input_batch.seq_lens
        assert (num_reqs, dcp_size, dcp_rank, interleave) == (1, 2, 1, 1)
        output[0] = 2

    monkeypatch.setattr(mrv2, "prepare_dcp_local_seq_lens", prepare_dcp)
    monkeypatch.setattr(mrv2, "build_slot_mappings_by_layer", lambda *args: object())

    class MetadataBuilt(Exception):
        pass

    def prepare_attn(batch, *args):
        events.append("prepare_attn")
        if expect_dcp:
            assert batch.dcp_local_seq_lens is not None
            assert batch.dcp_local_seq_lens.data_ptr() == dcp_local_seq_lens.data_ptr()
            assert batch.dcp_local_seq_lens.tolist() == [2]
        else:
            assert batch.dcp_local_seq_lens is None
        raise MetadataBuilt

    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = None
    runner.dp_size = 1
    runner.dp_rank = 0
    runner.use_dcp = True
    runner.vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4)
    )
    runner.input_buffers = SimpleNamespace(dcp_local_seq_lens=dcp_local_seq_lens)
    runner.dcp_size = 2
    runner.dcp_rank = 1
    runner.cp_interleave = 1
    runner.prepare_dummy_attn = lambda batch: (object(), object())
    runner.kv_cache_config = object()
    runner.attn_groups = object()
    runner.model_state = SimpleNamespace(prepare_attn=prepare_attn)

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"dummy": 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
    )
    with pytest.raises(MetadataBuilt):
        runner.execute_model(scheduler_output, dummy_run=True)

    assert events == ["prepare_dcp", "prepare_attn"]

    expect_dcp = False
    for flash_attn_version, is_sm90 in ((3, True), (4, False)):
        events.clear()
        input_batch.dcp_local_seq_lens = None
        runner.vllm_config.attention_config.flash_attn_version = flash_attn_version
        with pytest.raises(MetadataBuilt):
            runner.execute_model(scheduler_output, dummy_run=True)
        assert events == ["prepare_attn"]
