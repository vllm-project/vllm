# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3_5TextConfig

from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.model_executor.models import qwen3_5
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.mimo import MiMoModel
from vllm.models.minimax_m3.nvidia import model as minimax_m3
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    reserve_aux_intermediate_tensor_slots,
    verify_supports_aux_hidden_states_over_pp,
)
from vllm.v1.worker.gpu.spec_decode.extract_hidden_states_config import (
    ExtractHiddenStatesSpeculatorConfig,
)
from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig


@pytest.mark.parametrize(
    ("config_cls", "hf_config", "expected"),
    [
        (
            Eagle3SpeculatorConfig,
            {"eagle_config": {"use_aux_hidden_state": False}},
            False,
        ),
        (
            Eagle3SpeculatorConfig,
            {"eagle_config": {"use_aux_hidden_state": True}},
            True,
        ),
        (Eagle3SpeculatorConfig, {"eagle_config": {}}, True),
        (Eagle3SpeculatorConfig, {"eagle_config": None}, True),
        (Eagle3SpeculatorConfig, {}, True),
        (
            DFlashSpeculatorConfig,
            {"eagle_config": {"use_aux_hidden_state": False}},
            True,
        ),
        (
            DSparkSpeculatorConfig,
            {"eagle_config": {"use_aux_hidden_state": False}},
            True,
        ),
        (ExtractHiddenStatesSpeculatorConfig, {}, True),
        (MTPSpeculatorConfig, {}, False),
    ],
)
def test_aux_hidden_state_requirement_respects_eagle3_config(
    config_cls, hf_config, expected
):
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_config)),
    )
    assert config_cls(spec_config).uses_aux_hidden_states is expected


def test_aux_layers_are_sorted_and_deduplicated():
    model = EagleModelMixin()
    model._set_aux_hidden_state_layers((48, 3, 90, 24, 48))
    assert model.aux_hidden_state_layers == (3, 24, 48, 90)


def test_mimo_does_not_inherit_aux_hidden_state_pp_support():
    inner = MiMoModel.__new__(MiMoModel)
    target = SimpleNamespace(model=inner)

    assert not inner.supports_aux_hidden_states_over_pp
    with pytest.raises(ValueError, match="does not support eagle3"):
        verify_supports_aux_hidden_states_over_pp(target, "eagle3")


class _PPAuxLayer(torch.nn.Module):
    """Aliases and mutates residual in place like MiniMaxM3DecoderLayer.

    With ``deferred``, the FFN output is this rank's half of a 2-way TP sum, and
    the next input norm completes the all-reduce.
    """

    def __init__(self, index, deferred):
        super().__init__()
        self.index = index
        self.ffn_all_reduce_deferred = deferred
        self.fuse_input_allreduce = False

    def forward_input_norm(self, hidden_states, residual):
        if residual is None:
            return hidden_states, hidden_states
        if self.fuse_input_allreduce:
            hidden_states = hidden_states * 2
        residual.add_(hidden_states)
        return hidden_states, residual

    def forward_after_input_norm(self, positions, hidden_states, residual):
        residual.add_(self.index + 1)
        output = torch.full_like(hidden_states, 10 * (self.index + 1))
        return output / 2 if self.ffn_all_reduce_deferred else output, residual


class _PPAuxNorm(torch.nn.Module):
    """Fused add + norm (identity scaling) that overwrites its inputs in place."""

    def forward(self, hidden_states, residual):
        residual.add_(hidden_states)
        hidden_states.copy_(residual)
        return hidden_states, residual


@pytest.fixture
def minimax_pp_stage(monkeypatch):
    """Use the real MiniMax constructor/forward and aux helpers on CPU tensors."""
    group = SimpleNamespace(world_size=2, is_first_rank=True, is_last_rank=False)
    bounds = [0, 3]
    monkeypatch.setattr(minimax_m3, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        minimax_m3.MiniMaxM3Model, "_init_fused_moe_finalize", lambda *_: None
    )
    monkeypatch.setattr(
        minimax_m3,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        minimax_m3, "MiniMAXGemmaRMSNorm", lambda *args, **kwargs: _PPAuxNorm()
    )
    deferred = [False]
    monkeypatch.setattr(
        minimax_m3,
        "MiniMaxM3DecoderLayer",
        lambda **kwargs: _PPAuxLayer(
            int(kwargs["prefix"].rsplit(".", 1)[1]), deferred[0]
        ),
    )
    monkeypatch.setattr(
        minimax_m3,
        "fused_allreduce_gemma_rms_norm",
        lambda hidden_states, residual, norm: norm(hidden_states * 2, residual),
    )

    def make_layers(num_layers, build, prefix):
        return (
            *bounds,
            torch.nn.ModuleList(
                build(f"{prefix}.{index}")
                if bounds[0] <= index < bounds[1]
                else torch.nn.Identity()
                for index in range(num_layers)
            ),
        )

    monkeypatch.setattr(minimax_m3, "make_layers", make_layers)

    def make_stage(start, end, taps, world_size=2, deferred_all_reduce=False):
        bounds[:] = [start, end]
        deferred[0] = deferred_all_reduce
        group.world_size = world_size
        group.is_first_rank = start == 0
        group.is_last_rank = end == 6
        config = SimpleNamespace(
            vocab_size=16, hidden_size=6144, num_hidden_layers=6, rms_norm_eps=1e-6
        )
        model = minimax_m3.MiniMaxM3Model(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(hf_text_config=config),
                quant_config=None,
                speculative_config=None,
                kernel_config=SimpleNamespace(moe_backend="auto"),
                use_v2_model_runner=True,
            )
        )
        model._set_aux_hidden_state_layers(taps)
        target = SimpleNamespace(
            model=model,
            make_empty_intermediate_tensors=model.make_empty_intermediate_tensors,
        )
        reserve_aux_intermediate_tensor_slots(target)
        return model, target

    return make_stage


# Input 7; _PPAuxLayer i adds 11 * (i + 1) to hidden + residual.
def _expected_aux_value(tap):
    return 7 + 11 * tap * (tap + 1) // 2


def _pp_aux_state(value):
    return torch.full((2, 6144), float(value), device="cpu")


@pytest.mark.parametrize("taps", [(0, 3, 6), (0, 1, 3), (4, 5, 6)])
def test_minimax_pp2_preserves_ordered_global_aux_taps(minimax_pp_stage, taps):
    first, _ = minimax_pp_stage(0, 3, taps)
    sent = first(None, None, None, _pp_aux_state(7))
    last, target = minimax_pp_stage(3, 6, taps)
    reserved = target.make_empty_intermediate_tensors(2, torch.float32, "cpu")
    assert set(sent.tensors) == set(reserved.tensors)
    result = last(None, None, sent)
    assert isinstance(result, tuple)
    output, aux = result
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))
    for hidden, tap in zip(aux, taps, strict=True):
        torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))


def test_minimax_missing_upstream_aux_slot_is_not_silently_dropped(minimax_pp_stage):
    model, _ = minimax_pp_stage(3, 6, (0, 3, 6))
    incoming = IntermediateTensors(
        {
            "hidden_states": _pp_aux_state(30),
            "residual": _pp_aux_state(43),
            "aux_hidden_states_0": _pp_aux_state(7),
        }
    )
    with pytest.raises(RuntimeError, match="Missing aux_hidden_states_1"):
        model(None, None, incoming)


@pytest.mark.parametrize("taps", [(), (0, 3, 6)])
def test_minimax_pp1_output_and_aux_taps(minimax_pp_stage, taps):
    model, _ = minimax_pp_stage(0, 6, taps, world_size=1)
    output = model(None, None, None, _pp_aux_state(7))
    if taps:
        output, aux = output
        for hidden, tap in zip(aux, taps, strict=True):
            torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


def test_minimax_pp2_without_aux_keeps_hidden_and_residual_transport(minimax_pp_stage):
    first, _ = minimax_pp_stage(0, 3, ())
    sent = first(None, None, None, _pp_aux_state(7))
    assert set(sent.tensors) == {"hidden_states", "residual"}
    torch.testing.assert_close(sent["hidden_states"], _pp_aux_state(30))
    torch.testing.assert_close(sent["residual"], _pp_aux_state(43))
    last, target = minimax_pp_stage(3, 6, ())
    assert set(
        target.make_empty_intermediate_tensors(12, torch.bfloat16, "cpu").tensors
    ) == {"hidden_states", "residual"}
    output = last(None, None, sent)
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


@pytest.mark.parametrize("taps", [(0, 3, 6), (1, 2, 5)])
def test_minimax_aux_taps_follow_deferred_tp_all_reduce(minimax_pp_stage, taps):
    """Under TP each FFN output stays a partial sum until the next input norm or
    the final norm all-reduces it, so aux taps must read the reduced stream."""
    model, _ = minimax_pp_stage(0, 6, taps, world_size=1, deferred_all_reduce=True)
    assert model.fuse_final_norm_allreduce
    output, aux = model(None, None, None, _pp_aux_state(7))
    for hidden, tap in zip(aux, taps, strict=True):
        torch.testing.assert_close(hidden, _pp_aux_state(_expected_aux_value(tap)))
    torch.testing.assert_close(output, _pp_aux_state(_expected_aux_value(6)))


def _check_qwen3_5_aux_layout(rank, model_path, taps, implementation):
    from vllm.models.qwen3_5.nvidia import model as nvidia_qwen3_5

    model_cls = (
        nvidia_qwen3_5.Qwen3_5ForCausalLM
        if implementation == "nvidia"
        else qwen3_5.Qwen3_5ForCausalLM
    )

    config = VllmConfig(
        model_config=ModelConfig(
            model=model_path,
            skip_tokenizer_init=True,
            dtype="float32",
            max_model_len=32,
            enforce_eager=True,
        ),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=2, distributed_executor_backend="mp"
        ),
        compilation_config=CompilationConfig(mode=0),
    )
    try:
        with set_current_vllm_config(config):
            parallel_state.init_distributed_environment(
                world_size=2,
                rank=rank,
                local_rank=rank,
                backend="gloo",
                distributed_init_method=f"file://{model_path}/store",
                timeout=timedelta(seconds=60),
            )
            parallel_state.initialize_model_parallel(1, 2, backend="gloo")
            target = model_cls(vllm_config=config)
            target.set_aux_hidden_state_layers(())
            reserve_aux_intermediate_tensor_slots(target)
            buffers = target.make_empty_intermediate_tensors(2, torch.float32, "cpu")
            assert set(buffers.tensors) == {"hidden_states", "residual"}
            assert target.model.pack_local_aux_hidden_states([]) == {}
            assert target.model.collect_remote_aux_hidden_states(None) == []

            target.set_aux_hidden_state_layers(taps)
            model = target.model
            assert model.aux_hidden_state_layers == (0, 2, 4)
            assert (model.start_layer, model.end_layer) == (rank * 2, rank * 2 + 2)
            verify_supports_aux_hidden_states_over_pp(target, "dflash")
            local_taps = (0, 2) if rank == 0 else (4,)
            local = [torch.full((2, 128), float(tap)) for tap in local_taps]
            packed = model.pack_local_aux_hidden_states(local)
            slots = (0, 1) if rank == 0 else (2,)
            assert tuple(packed) == tuple(f"aux_hidden_states_{i}" for i in slots)
            for tensor, tap in zip(packed.values(), local_taps, strict=True):
                torch.testing.assert_close(
                    tensor, torch.full((2, 128), float(tap)), rtol=0, atol=0
                )
            if rank == 1:
                reserve_aux_intermediate_tensor_slots(target)
                buffers = target.make_empty_intermediate_tensors(
                    2, torch.float32, "cpu"
                )
                assert set(buffers.tensors) == {
                    "hidden_states",
                    "residual",
                    "aux_hidden_states_0",
                    "aux_hidden_states_1",
                }
                for key in ("aux_hidden_states_0", "aux_hidden_states_1"):
                    assert buffers[key].shape == (2, 128)
                incoming = IntermediateTensors(
                    {
                        f"aux_hidden_states_{i}": torch.full((2, 128), float(tap))
                        for i, tap in enumerate((0, 2))
                    }
                )
                received = model.collect_remote_aux_hidden_states(incoming)
                assert len(received) == 2
                for tensor, tap in zip(received, (0, 2), strict=True):
                    torch.testing.assert_close(
                        tensor, torch.full((2, 128), float(tap)), rtol=0, atol=0
                    )
                incomplete = IntermediateTensors(
                    {"aux_hidden_states_0": incoming["aux_hidden_states_0"]}
                )
                with pytest.raises(RuntimeError, match="Missing aux_hidden_states_1"):
                    model.collect_remote_aux_hidden_states(incomplete)
    finally:
        parallel_state.cleanup_dist_env_and_memory()


@pytest.mark.cpu_test
@pytest.mark.parametrize("implementation", ["legacy", "nvidia"])
@pytest.mark.parametrize("taps", [(0, 2, 4), (4, 0, 2, 4)])
def test_qwen3_5_declares_and_configures_aux_hidden_state_pp(
    tmp_path, taps, implementation
):
    """The public setter must initialize global auxiliary slots on both stages."""
    Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"] * 2,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    context = torch.multiprocessing.spawn(
        _check_qwen3_5_aux_layout,
        args=(str(tmp_path), taps, implementation),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + 120
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Qwen3.5 auxiliary-layout workers timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.is_alive():
                raise RuntimeError(
                    f"Failed to stop auxiliary-layout worker {process.pid}"
                )
