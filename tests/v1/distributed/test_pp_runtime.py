# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-adapter contract with real events and dictionary transport."""

from dataclasses import replace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from transformers import LlamaConfig

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.worker.pp.interface import PPLayerSpec
from vllm.v1.worker.pp.runtime import PPRuntime


class LayerChain(torch.nn.Module):
    def __init__(self, rank, device):
        super().__init__()
        torch.manual_seed(7)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(12)])
        self.execution_range = (0, 6) if rank == 0 else (6, 12)
        self.to(device)

    def get_dynamic_pp_layer_spec(self):
        return PPLayerSpec(6, 12)

    def set_dynamic_pp_layer_range(self, start, end):
        self.execution_range = start, end

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.empty(
                    batch_size, 4, 64, dtype=dtype, device=device
                )
            }
        )

    def forward(self, value):
        start, end = self.execution_range
        for layer in self.layers[start:end]:
            value = torch.tanh(layer(value))
        return value


def fresh_schedule(rows):
    request = NewRequestData(
        req_id="request",
        prompt_token_ids=[1] * rows,
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )
    return replace(
        SchedulerOutput.make_empty(),
        scheduled_new_reqs=[request],
        num_scheduled_tokens={"request": rows},
        total_num_scheduled_tokens=rows,
    )


def make_config(path, offset, device_type):
    return VllmConfig(
        model_config=ModelConfig(
            model=path,
            skip_tokenizer_init=True,
            enforce_eager=True,
            max_model_len=256,
            dtype="float32",
        ),
        device_config=DeviceConfig(device=device_type),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=2, distributed_executor_backend="mp"
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=256,
            is_encoder_decoder=False,
            max_num_batched_tokens=256,
            enable_chunked_prefill=False,
        ),
        cache_config=CacheConfig(enable_prefix_caching=False),
        additional_config={
            "pp_send_buffer_size": 2,
            "dynamic_pp_config": {
                "enabled": True,
                "redundant_layers": 2,
                "initial_offset": offset,
                "adaptive": False,
            },
        },
    )


@torch.inference_mode()
def runtime_worker(rank, rendezvous, first_two_posted, path, offset):
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    config = make_config(path, offset, "cuda")
    with set_current_vllm_config(config):
        pp = GroupCoordinator([[0, 1]], rank, "nccl", False, group_name="pp")
        tp = GroupCoordinator([[0], [1]], rank, "nccl", False, group_name="tp")
        model = LayerChain(rank, device)
        runtime = PPRuntime(config, model, device, torch.cuda, pp, tp)
    try:
        # Establish the transport before deliberately delaying the consumer.
        if rank == 0:
            pp.send_tensor_dict({"hidden_states": torch.zeros(1, device=device)})
        else:
            pp.recv_tensor_dict()
        torch.accelerator.synchronize()
        pp.barrier()
        for batch in range(12):
            rows = 64 if batch % 2 else 128
            schedule = fresh_schedule(rows)
            if rank == 1 and batch == 0:
                assert first_two_posted.wait(45), "second batch waited for its receiver"
            runtime.prepare(schedule)
            if rank == 0:
                value = torch.full(
                    (rows, 4, 64), batch, device=device, dtype=torch.float32
                )
            else:
                tensors, handles, callbacks = pp.irecv_tensor_dict(all_gather_group=tp)
                assert tensors is not None
                runtime.receive_header(tensors)
                for handle in handles:
                    handle.wait()
                for callback in callbacks:
                    callback()
                value = tensors["hidden_states"]
            runtime.before_forward(schedule)
            output = model(value)
            if rank == 0:
                runtime.send({"hidden_states": output})
                output.fill_(-100)
                if batch == 1:
                    first_two_posted.set()
            else:
                runtime.sampled()
                reference = torch.full_like(value, batch)
                for layer in model.layers:
                    reference = torch.tanh(layer(reference))
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
        runtime.drain()
        torch.accelerator.synchronize()
        feedback = runtime.feedback
        assert feedback is not None
        feedback.poll()
        pp.barrier()
        feedback.poll()
        stats = runtime.stats()
        assert stats["batches"] == 12 and stats["boundary"] == 6 + offset
        assert stats["device_feedback"]["completed_timing_samples"] == 12
        if rank == 0:
            assert stats["buffer_sends"] == 12
            samples = feedback.take_samples()
            assert samples[-1].batch_id == 11
            assert feedback.num_paired_samples == 12
            assert feedback.queue_depth == 0
    finally:
        pp.destroy()
        tp.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("offset", [-2, 0, 2])
def test_runtime_preserves_outputs_and_pairs_completed_timings(
    tmp_path, monkeypatch, offset
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config_path = tmp_path / "model"
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=12,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
    ).save_pretrained(config_path)
    context = torch.multiprocessing.get_context("spawn")
    torch.multiprocessing.spawn(
        runtime_worker,
        args=(
            f"file://{tmp_path / 'rendezvous'}",
            context.Event(),
            str(config_path),
            offset,
        ),
        nprocs=2,
        join=True,
    )
