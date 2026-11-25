# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.multiprocessing as mp
from transformers import Qwen3Config, Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config import (
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    reserve_aux_intermediate_tensor_slots,
)


def _write_tiny_qwen4_exp_config(tmp_path, wrapped: bool = False) -> None:
    hf_config = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        layer_types=["full_attention"] * 4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        eos_token_id=1,
    )
    if wrapped:
        hf_config = Qwen4ExpConfig(
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=hf_config,
        )
    else:
        hf_config.architectures = ["Qwen4ExpForCausalLM"]
    hf_config.save_pretrained(tmp_path)


def _make_vllm_config(
    model_path: str, pp_size: int = 1, wrapped: bool = False
) -> VllmConfig:
    model_config = ModelConfig(
        model=model_path,
        skip_tokenizer_init=True,
        max_model_len=32,
        dtype="float32",
        language_model_only=wrapped,
    )
    return VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(pipeline_parallel_size=pp_size),
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_nvidia_qwen4_exp_selects_contracted_aux_capture_layers(
    tmp_path, wrapped: bool
) -> None:
    """Select the real decoder layers whose HC attention inputs feed a drafter."""
    _write_tiny_qwen4_exp_config(tmp_path, wrapped=wrapped)
    vllm_config = _make_vllm_config(str(tmp_path), wrapped=wrapped)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'dist'}",
            backend="gloo",
        )
        try:
            initialize_model_parallel(backend="gloo")
            with torch.device("meta"):
                model = (
                    Qwen4ExpForConditionalGeneration(vllm_config=vllm_config)
                    if wrapped
                    else Qwen4ExpForCausalLM(vllm_config=vllm_config)
                )
            model.set_aux_hidden_state_layers((4, 2, 1, 0))
            decoder = model.language_model.model if wrapped else model.model
            assert decoder.aux_hidden_state_layers == (0, 1, 2, 4)
            assert [layer.capture_aux_hidden_state for layer in decoder.layers] == [
                False,
                True,
                True,
                False,
            ]
            with pytest.raises(ValueError, match="valid layer indices"):
                model.set_aux_hidden_state_layers((5,))
        finally:
            cleanup_dist_env_and_memory()


def _check_pp_aux_slots(rank: int, model_path: str, port: int, wrapped: bool) -> None:
    vllm_config = _make_vllm_config(model_path, pp_size=2, wrapped=wrapped)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        try:
            initialize_model_parallel(pipeline_model_parallel_size=2, backend="gloo")
            with torch.device("meta"):
                model = (
                    Qwen4ExpForConditionalGeneration(vllm_config=vllm_config)
                    if wrapped
                    else Qwen4ExpForCausalLM(vllm_config=vllm_config)
                )
            model.set_aux_hidden_state_layers((0, 2, 3, 4))
            reserve_aux_intermediate_tensor_slots(model)
            inner = model.language_model.model if wrapped else model.model
            assert (inner.start_layer, inner.end_layer) == (
                (0, 2) if rank == 0 else (2, 4)
            )
            upstream_keys, local_keys = inner.aux_hidden_state_slot_keys()
            assert upstream_keys == (() if rank == 0 else ("aux_hidden_states_0",))
            assert local_keys == (("aux_hidden_states_0",) if rank == 0 else ())
            slots = model.make_empty_intermediate_tensors(
                1, torch.float32, torch.device("cpu")
            )
            expected_keys = {"hidden_states"}
            if rank == 1:
                expected_keys.add("aux_hidden_states_0")
            assert set(slots.tensors) == expected_keys
            if rank == 0:
                assert not inner.layers[0].capture_aux_hidden_state
                assert not inner.layers[1].capture_aux_hidden_state
                # The first slot accepts a native-width embedding-shaped tensor.
                packed = inner.pack_local_aux_hidden_states([torch.ones(1, 128)])
                assert set(packed) == {"aux_hidden_states_0"}
                torch.distributed.send(packed["aux_hidden_states_0"], dst=1)
            else:
                assert inner.layers[2].capture_aux_hidden_state
                assert inner.layers[3].capture_aux_hidden_state
                received = torch.empty(1, 128)
                torch.distributed.recv(received, src=0)
                incoming = IntermediateTensors(
                    {
                        "hidden_states": torch.zeros(1, 256),
                        "aux_hidden_states_0": received,
                    }
                )
                local = [torch.full((1, 128), value) for value in (2.0, 3.0, 4.0)]
                assert set(inner.pack_local_aux_hidden_states(local)) == {
                    "aux_hidden_states_1",
                    "aux_hidden_states_2",
                    "aux_hidden_states_3",
                }
                for index, state in enumerate(
                    [incoming[key] for key in upstream_keys] + local, start=1
                ):
                    torch.testing.assert_close(
                        state, torch.full((1, 128), float(index))
                    )
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize("wrapped", [False, True])
def test_nvidia_qwen4_exp_relays_aux_slot_order_across_pp(
    tmp_path,
    wrapped: bool,
) -> None:
    """PP slot keys and synthetic tensors retain their order across ranks."""
    _write_tiny_qwen4_exp_config(tmp_path, wrapped=wrapped)
    mp.spawn(
        _check_pp_aux_slots,
        args=(str(tmp_path), get_open_port(), wrapped),
        nprocs=2,
    )


def _check_pp_target_embedding(
    rank: int,
    model_path: str,
    draft_path: str,
    port: int,
    wrapped: bool,
    use_dflash: bool,
) -> None:
    vllm_config = _make_vllm_config(model_path, pp_size=2, wrapped=wrapped)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        try:
            initialize_model_parallel(pipeline_model_parallel_size=2, backend="gloo")
            if use_dflash:
                vllm_config.speculative_config = SpeculativeConfig(
                    method="dflash",
                    model=draft_path,
                    num_speculative_tokens=2,
                    target_model_config=vllm_config.model_config,
                    target_parallel_config=vllm_config.parallel_config,
                )
                spec_config = vllm_config.speculative_config
                assert spec_config.draft_model_config.may_use_target_embed
            with torch.device("meta"):
                model = (
                    Qwen4ExpForConditionalGeneration(vllm_config=vllm_config)
                    if wrapped
                    else Qwen4ExpForCausalLM(vllm_config=vllm_config)
                )
            inner = model.language_model.model if wrapped else model.model
            has_embedding = rank == 0 or use_dflash
            assert (
                isinstance(inner.embed_tokens, VocabParallelEmbedding) == has_embedding
            )
            assert isinstance(inner.embed_tokens, PPMissingLayer) != has_embedding
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("use_dflash", [False, True])
def test_nvidia_qwen4_exp_keeps_target_embedding_for_pp_dflash(
    tmp_path, wrapped: bool, use_dflash: bool
) -> None:
    """The last PP rank supplies a shared embedding only when the draft needs it."""
    _write_tiny_qwen4_exp_config(tmp_path, wrapped=wrapped)
    draft_path = tmp_path / "draft"
    Qwen3Config(
        architectures=["DFlash2DraftModel"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        dflash_config={"target_layer_ids": [0], "block_size": 2},
    ).save_pretrained(draft_path)
    mp.spawn(
        _check_pp_target_embedding,
        args=(str(tmp_path), str(draft_path), get_open_port(), wrapped, use_dflash),
        nprocs=2,
    )
