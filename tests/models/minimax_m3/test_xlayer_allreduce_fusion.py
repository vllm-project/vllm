# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 (AMD) defers each FFN all-reduce into the next norm.

Every deferred all-reduce must be finished exactly once, by the next layer's
input norm (bf16 or MXFP8) or the final norm; each EAGLE3 capture must be
recorded once and never as a per-rank partial sum; and mono decode, whose
kernels expect reduced layer outputs, must turn deferral off exactly when it
can run. At TP2 the deferred model's final hidden states and captures must
match the model that reduces every FFN.
"""

import os
import zlib

import pytest
import ray
import torch
from transformers import MiniMaxM3VLConfig

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from ...utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="MiniMax-M3 AMD model"
)

LAYERS = 5
DENSE = 2
# Capture 0 is the embedding and capture LAYERS the final norm's residual.
AUX_LAYERS = tuple(range(LAYERS + 1))
_CKPT_ENV = "VLLM_TEST_M3_XLAYER_CKPT"
_MXFP8_ENV = "VLLM_TEST_M3_XLAYER_MXFP8"


def _save_checkpoint(path, quantization_config=None) -> str:
    no_sparse = [0] * LAYERS
    quant = (
        {}
        if quantization_config is None
        else {"quantization_config": quantization_config}
    )
    MiniMaxM3VLConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=256,
            intermediate_size=128,
            dense_intermediate_size=256,
            num_hidden_layers=LAYERS,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            rotary_dim=64,
            num_local_experts=4,
            num_experts_per_tok=2,
            moe_layer_freq=[0] * DENSE + [1] * (LAYERS - DENSE),
            sparse_attention_config=dict(
                use_sparse_attention=True,
                sparse_index_dim=64,
                sparse_num_index_heads=1,
                sparse_topk_blocks=4,
                sparse_block_size=128,
                sparse_disable_index_value=no_sparse,
                sparse_score_type="max",
                sparse_init_block=0,
                sparse_local_block=1,
                sparse_attention_freq=no_sparse,
            ),
        ),
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
        **quant,
    ).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _save_checkpoint(tmp_path_factory.mktemp("minimax_m3_tiny"))


@pytest.fixture(scope="module")
def mxfp8_checkpoint(tmp_path_factory):
    # An MXFP8 qkv_proj sends every input norm down its MXFP8 branch. The FFN
    # stays bf16: these shapes are below the MXFP8 MoE kernels' minimum.
    return _save_checkpoint(
        tmp_path_factory.mktemp("minimax_m3_tiny_mxfp8"),
        {
            "quant_method": "mxfp8",
            "ignored_layers": ["lm_head", "mlp", "block_sparse_moe"],
        },
    )


class _UnspecializedDecode:
    """Runs the specialized decode layer loop with no layer specialized."""

    def plan(self, positions, intermediate_tensors):
        return None

    def after_step(self, step, failure=None):
        if failure is not None:
            raise failure


def _init_rank(tp_size, pp_size, rank, distributed_init_port) -> torch.device:
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    ensure_model_parallel_initialized(tp_size, pp_size)
    init_workspace_manager(device)
    return device


def _build_random_model(device, rank, pipeline_parallel_size=1):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model

    model_config = ModelConfig(os.environ[_CKPT_ENV], max_model_len=1024)
    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.parallel_config.pipeline_parallel_size = pipeline_parallel_size
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        device,
    ):
        model = MiniMaxM3Model(vllm_config=vllm_config, prefix="model")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 0:  # attention q/k/v scales keep their defaults
                continue
            if p.dtype == torch.uint8:  # E8M0 block scales; 127 is 2**0
                p.fill_(127)
                continue
            g = torch.Generator(device=device).manual_seed(
                zlib.crc32(f"{name}/{rank}".encode())
            )
            std = 0.1 if p.ndim == 1 else p.shape[-1] ** -0.5
            p.copy_(torch.randn(p.shape, generator=g, device=device) * std)
    process_weights_after_loading(model, model_config, device)
    model._set_aux_hidden_state_layers(AUX_LAYERS)
    return model, vllm_config


def _record_input_norms(model) -> None:
    """Record every layer's `forward_input_norm` output on
    `model.input_norm_outputs`."""
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )

    for layer in model.layers:

        def record(hidden_states, residual, norm=layer.forward_input_norm):
            out, residual = norm(hidden_states, residual)
            model.input_norm_outputs.append(
                QuantizedActivation(
                    out.data.clone(),
                    out.scale.clone(),
                    out.orig_dtype,
                    out.orig_shape,
                    out.quant_key,
                )
                if isinstance(out, QuantizedActivation)
                else out.clone()
            )
            return out, residual

        layer.forward_input_norm = record


def _dequant(x) -> torch.Tensor:
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    if isinstance(x, QuantizedActivation):
        return dequant_mxfp8_to_bf16(x.data, x.scale)
    return x


@ray.remote(num_gpus=1, max_calls=1)
def _run_deferred_matches_reduced(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, distributed_init_port
):
    from vllm.forward_context import set_forward_context

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = _init_rank(tp_size, pp_size, rank, distributed_init_port)
        input_ids = torch.arange(37, device=device) * 7 % 256
        positions = torch.arange(37, device=device)

        def run(model, vllm_config):
            model.input_norm_outputs = []
            with torch.inference_mode(), set_forward_context(None, vllm_config):
                out, aux = model(input_ids, positions, None)
                assert len(aux) == len(AUX_LAYERS)
                torch.testing.assert_close(aux[0], model.embed_input_ids(input_ids))
            assert len(model.input_norm_outputs) == LAYERS
            return [out, *aux]

        deferred, deferred_config = _build_random_model(device, rank)
        _record_input_norms(deferred)
        assert deferred.fuse_final_norm_allreduce
        mxfp8 = os.environ[_MXFP8_ENV] == "1"
        assert all(layer.fuse_input_norm_mxfp8 is mxfp8 for layer in deferred.layers)
        got = run(deferred, deferred_config)

        # PP > 1 turns deferral off. Faking it in the config builds the
        # reduce-every-FFN reference, which still runs as a single stage.
        reduced, reduced_config = _build_random_model(
            device, rank, pipeline_parallel_size=2
        )
        _record_input_norms(reduced)
        assert not reduced.fuse_final_norm_allreduce
        ref = run(reduced, reduced_config)
        # Without AITER the deferred reduce is the same all-reduce, one call
        # later, so the outputs match bitwise. AITER's fused kernels may sum
        # in another order.
        tol = (
            {"rtol": 2e-2, "atol": 2e-2}
            if os.environ["VLLM_ROCM_USE_AITER"] == "1"
            else {"rtol": 0, "atol": 0}
        )
        for a, b in zip(got, ref, strict=True):
            torch.testing.assert_close(a, b, **tol)
        # Attention sees no metadata and returns zeros, so the input norms'
        # outputs (bf16, or MXFP8 for qkv_proj) reach nothing compared above:
        # compare them directly. Without AITER the deferred reduce feeds the
        # same norm kernel; AITER's fused all-reduce + norm has no MXFP8
        # output, so there a bf16 result meets an MXFP8 one (one e4m3 step).
        aiter = os.environ["VLLM_ROCM_USE_AITER"] == "1"
        for a, b in zip(
            deferred.input_norm_outputs, reduced.input_norm_outputs, strict=True
        ):
            if not aiter:
                assert type(a) is type(b)
            mixed = type(a) is not type(b)
            torch.testing.assert_close(
                _dequant(a),
                _dequant(b),
                **({"rtol": 0.1, "atol": 0.05} if mixed else tol),
            )

        # The specialized decode loop captures after each layer instead.
        reduced.specialized_decode = _UnspecializedDecode()
        for a, b in zip(run(reduced, reduced_config), ref, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("use_aiter", [False, True])
def test_deferred_matches_reduced_at_tp2(
    checkpoint, mxfp8_checkpoint, monkeypatch, use_aiter, mxfp8
):
    """A dropped or doubled deferred reduce is off by a TP factor, and a
    doubled or missing capture shifts every later one."""
    pytest.importorskip("flydsl")
    tp_size = 2
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs.")
    if mxfp8 and not current_platform.supports_mx():
        pytest.skip("The MXFP8 input norm needs native MX (gfx95x).")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
    monkeypatch.setenv(_CKPT_ENV, mxfp8_checkpoint if mxfp8 else checkpoint)
    monkeypatch.setenv(_MXFP8_ENV, "1" if mxfp8 else "0")
    multi_process_parallel(monkeypatch, tp_size, 1, _run_deferred_matches_reduced)


@ray.remote(num_gpus=1, max_calls=1)
def _run_pp2_captures(
    monkeypatch: pytest.MonkeyPatch, tp_size, pp_size, rank, distributed_init_port
):
    from vllm.distributed import get_pp_group
    from vllm.forward_context import set_forward_context
    from vllm.sequence import IntermediateTensors

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        device = _init_rank(tp_size, pp_size, rank, distributed_init_port)
        model, vllm_config = _build_random_model(
            device, rank, pipeline_parallel_size=pp_size
        )
        positions = torch.arange(37, device=device)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            hidden = torch.randn(37, 256, device=device, dtype=torch.bfloat16)
            intermediate_tensors = IntermediateTensors(
                {"hidden_states": hidden, "residual": hidden.clone()}
            )
        with torch.inference_mode(), set_forward_context(None, vllm_config):
            out = model(positions * 7 % 256, positions, intermediate_tensors)
        if get_pp_group().is_last_rank:
            # This stage's start_layer capture belongs to the stage before.
            _, aux = out
            assert len(aux) == sum(i > model.start_layer for i in AUX_LAYERS)
        else:
            _, sends = model.aux_hidden_state_slot_keys()
            assert set(sends) == set(out.tensors) - {"hidden_states", "residual"}


def test_pp2_stage_boundary_captured_once(checkpoint, monkeypatch):
    pytest.importorskip("flydsl")
    pp_size = 2
    if torch.accelerator.device_count() < pp_size:
        pytest.skip(f"Need at least {pp_size} GPUs.")
    monkeypatch.setenv(_CKPT_ENV, checkpoint)
    multi_process_parallel(monkeypatch, 1, pp_size, _run_pp2_captures)


@pytest.mark.parametrize(
    ("parallel", "mono_runs"),
    [
        ({"tensor_parallel_size": 4}, True),
        ({"tensor_parallel_size": 2}, False),
        ({"tensor_parallel_size": 4, "enable_expert_parallel": True}, False),
    ],
)
def test_deferral_is_off_exactly_when_mono_decode_can_run(
    dist_init, checkpoint, monkeypatch, parallel, mono_runs
):
    pytest.importorskip("flydsl")
    from vllm.config import (
        CUDAGraphMode,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model
    from vllm.models.minimax_m3.amd.mono.dispatch import deployment_refusal
    from vllm.models.minimax_m3.amd.mono.envs import envs

    monkeypatch.setattr(envs, "MONO_ENABLE", True)
    vllm_config = VllmConfig(model_config=ModelConfig(checkpoint, max_model_len=1024))
    # A deployment mono serves, apart from the parallel layout under test.
    vllm_config.cache_config.cache_dtype = "fp8"
    vllm_config.cache_config.block_size = 128
    vllm_config.attention_config.indexer_kv_dtype = "fp8"
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    for key, val in parallel.items():
        setattr(vllm_config.parallel_config, key, val)
    assert (deployment_refusal(vllm_config) is None) is mono_runs
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        model = MiniMaxM3Model(vllm_config=vllm_config, prefix="model")
    # Under EP the MoE layers may reduce through all2all; the dense ones decide.
    assert all(
        layer.ffn_all_reduce_deferred is not mono_runs for layer in model.layers[:DENSE]
    )
    assert model.layers[1].fuse_input_allreduce is not mono_runs


def test_monokernel_turns_deferral_off(dist_init, checkpoint, monkeypatch):
    """--monokernel picks its kernels only after the layers exist, so the flag
    itself must keep every FFN all-reduce in its layer."""
    pytest.importorskip("flydsl")
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.models.minimax_m3.amd import model as m3

    # As if the specialized decode config were not known yet at layer build.
    monkeypatch.setattr(m3, "specialized_decode_config", lambda _: None)
    vllm_config = VllmConfig(model_config=ModelConfig(checkpoint, max_model_len=1024))
    vllm_config.kernel_config.monokernel = True
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        model = m3.MiniMaxM3Model(vllm_config=vllm_config, prefix="model")
    assert not any(layer.ffn_all_reduce_deferred for layer in model.layers)
    assert not model.fuse_final_norm_allreduce
