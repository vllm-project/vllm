# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib

import pytest
import torch
from safetensors.torch import save_file

from tests.v1.core.utils import new_kv_cache_group
from vllm.config import MambaConfig, VllmConfig
from vllm.config.mamba import SpareConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.layers.mamba.spare import load_bases, pack_basis, project_bc
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.worker.utils import allocate_kv_cache

REVISION = "a" * 40


def _write_basis(path, model, basis, method="state_pca"):
    heads, native_dim, rank = basis.shape
    save_file(
        {"layer.0": basis},
        path,
        metadata={
            "format": "spare-mamba2-v1",
            "method": method,
            "model": model,
            "revision": REVISION,
            "native_dim": str(native_dim),
            "rank": str(rank),
            "heads": str(heads),
        },
    )
    return {
        "basis_path": str(path),
        "basis_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rank": rank,
        "method": method,
    }


@pytest.mark.parametrize("rank", [16, 32, 64, 128])
def test_spare_projection_and_recurrence(rank):
    generator = torch.Generator().manual_seed(17)
    basis = (
        torch.eye(128).expand(4, -1, -1)
        if rank == 128
        else torch.linalg.qr(torch.randn(4, 128, rank, generator=generator)).Q
    )
    native = torch.randn(2, 3, 2, 128, generator=generator).bfloat16()
    packed = pack_basis(basis, groups=2)
    b_input, c_input = native.unbind()
    b, c = project_bc(b_input, c_input, packed, rank)
    b_native, c_native = native.float().repeat_interleave(2, dim=2).unbind()
    b_ref, c_ref = torch.einsum(
        "kthn,hnr->kthr", torch.stack((b_native, c_native)), basis
    )
    torch.testing.assert_close(b, b_ref.bfloat16())
    torch.testing.assert_close(c, c_ref.bfloat16())

    # Check closure on matched recurrent inputs before the BF16 projection cast.
    dense = torch.randn(4, 8, 128, generator=generator)
    compact = dense @ basis
    for t, x in enumerate(torch.randn(3, 4, 8, generator=generator)):
        dense = 0.7 * dense + x[..., None] * b_native[t, :, None, :]
        compact = 0.7 * compact + x[..., None] * b_ref[t, :, None, :]
        torch.testing.assert_close(compact, dense @ basis, atol=1e-5, rtol=1e-5)
        if rank == 128:
            torch.testing.assert_close(
                (compact * c_ref[t, :, None, :]).sum(-1),
                (dense * c_native[t, :, None, :]).sum(-1),
                atol=1e-5,
                rtol=1e-5,
            )


def test_spare_artifact_validation(tmp_path):
    path = tmp_path / "basis.safetensors"
    basis = torch.eye(32).repeat(2, 1, 1)
    fields = _write_basis(path, "checkpoint", basis, "identity")

    def load(digest, model="checkpoint"):
        return load_bases(
            str(path), digest, "identity", model, REVISION, (0,), 2, 32, 32
        )

    torch.testing.assert_close(load(fields["basis_sha256"])[0], basis)
    with pytest.raises(ValueError, match="SHA-256"):
        load("0" * 64)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load(fields["basis_sha256"], model="different-checkpoint")
    fields = _write_basis(path, "checkpoint", basis * 2, "identity")
    with pytest.raises(ValueError, match="Nonorthonormal"):
        load(fields["basis_sha256"])
    fields = _write_basis(path, "checkpoint", -basis, "identity")
    with pytest.raises(ValueError, match="identity SPARE basis"):
        load(fields["basis_sha256"])


def test_spare_temporal_shape_and_cache_backing():
    args = dict(
        tp_world_size=1,
        intermediate_size=256,
        n_groups=2,
        num_heads=4,
        head_dim=64,
        state_size=128,
        conv_kernel=4,
    )
    dense_conv, dense_state = MambaStateShapeCalculator.mamba2_state_shape(**args)
    spare_conv, spare_state = MambaStateShapeCalculator.mamba2_state_shape(
        **args, temporal_state_size=32
    )
    assert dense_conv == spare_conv
    assert dense_state == (4, 64, 128)
    assert spare_state == (4, 64, 32)

    config = VllmConfig()
    config.cache_config.kv_cache_layout = "BLHNC"
    attention = AttentionSpec(
        block_size=16, num_kv_heads=4, head_size=32, dtype=torch.bfloat16
    )
    allocation = []
    for shapes in ((dense_conv, dense_state), (spare_conv, spare_state)):
        mamba = MambaSpec(
            block_size=128, shapes=shapes, dtypes=(torch.bfloat16, torch.float32)
        )
        groups = [
            new_kv_cache_group(["attention"], attention),
            new_kv_cache_group(["mamba"], mamba),
        ]
        plan = get_kv_cache_config_from_groups(config, groups, num_blocks=32)
        caches = allocate_kv_cache(
            plan,
            torch.device("cpu"),
            config.cache_config.get_resolved_kv_cache_layout(),
        )
        allocation.append(caches["mamba"].untyped_storage().nbytes())
    assert allocation[0] - allocation[1] == 32 * 4 * 64 * (128 - 32) * 4


def test_spare_default_and_unsupported_model():
    assert VllmConfig().mamba_config.spare is None
    spare = SpareConfig(rank=32, basis_path="/basis.safetensors", basis_sha256="0" * 64)
    with pytest.raises(ValueError, match="requires the native Granite"):
        VllmConfig(mamba_config=MambaConfig(spare=spare))


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="SPARE requires NVIDIA CUDA."
)
def test_spare_native_ssd_to_ssu():
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
    from vllm.model_executor.layers.mamba.ops.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )
    from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(19)
    rank, heads, width = 32, 4, 64
    native = torch.randn(2, 28, 1, 128, device=device, generator=generator) * 0.1
    basis = torch.linalg.qr(
        torch.randn(heads, 128, rank, device=device, generator=generator)
    ).Q
    b_input, c_input = native.unbind()
    b, c = project_bc(b_input, c_input, pack_basis(basis, groups=1), rank)
    x = torch.randn(28, heads, width, device=device, generator=generator) * 0.1
    a = -torch.ones(heads, device=device)
    dt = torch.full((28, heads), 0.5, device=device)
    bias = torch.zeros_like(a)
    d = torch.ones_like(a)
    slots = torch.tensor([4, 1], device=device, dtype=torch.int32)
    cache = torch.zeros(5, heads, width, rank, device=device)
    cache[4].fill_(0.01)
    out = torch.empty_like(x)
    ref_cache = cache.clone()
    ref_out = torch.empty_like(out)
    for slot, start, end, decode in ((4, 0, 19, 27), (1, 19, 26, 26)):
        state = ref_cache[slot]
        for t in (*range(start, end), decode):
            state.mul_(torch.exp(a * dt[t])[:, None, None]).add_(
                dt[t, :, None, None] * x[t, :, :, None] * b[t, :, None, :]
            )
            ref_out[t] = (state * c[t, :, None, :]).sum(-1) + x[t]

    cu = torch.tensor([0, 19, 26], device=device, dtype=torch.int32)
    chunks, last, seq = compute_varlen_chunk_metadata(cu, 16)
    mamba_chunk_scan_combined_varlen(
        x[:26],
        dt[:26],
        a,
        b[:26],
        c[:26],
        16,
        cu,
        chunks,
        last,
        seq,
        out[:26],
        D=d,
        dt_bias=bias,
        initial_states=cache[slots],
        state_dtype=torch.float32,
        final_states_out=cache,
        final_state_indices=slots,
    )
    # Decode the same requests in reverse order after the indexed SSD store.
    selective_state_update(
        cache,
        x[26:],
        dt[26:, :, None].expand(-1, -1, width),
        a[:, None, None].expand(-1, width, rank),
        b[26:],
        c[26:],
        d[:, None].expand(-1, width),
        bias[:, None].expand(-1, width),
        state_batch_indices=slots.flip(0),
        dst_state_batch_indices=slots.flip(0),
        out=out[26:],
    )
    torch.testing.assert_close(cache, ref_cache, atol=2e-4, rtol=1e-3)
    torch.testing.assert_close(out, ref_out, atol=2e-4, rtol=1e-3)


def _state_allocation(model):
    conv, temporal = model.model.layers[0].mamba.kv_cache
    assert conv.is_cuda and temporal.is_cuda and temporal.dtype == torch.float32
    assert conv.untyped_storage().data_ptr() == temporal.untyped_storage().data_ptr()
    return tuple(conv.shape), tuple(temporal.shape), temporal.untyped_storage().nbytes()


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="SPARE requires NVIDIA CUDA."
)
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "decode_graph"])
def test_spare_granite_serving(tmp_path, vllm_runner, enable_pickle, graph):
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

    from vllm import SamplingParams

    model_path = str(tmp_path / "granite")
    config = GraniteMoeHybridConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["linear_attention", "full_attention"],
        mamba_n_heads=4,
        mamba_n_groups=1,
        mamba_d_head=64,
        mamba_d_state=128,
        max_position_embeddings=128,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        GraniteMoeHybridForCausalLM(config).save_pretrained(model_path)

    allocation = []
    generations = []
    for rank in (None, 128, 32):
        spare = None
        if rank is not None:
            basis = torch.eye(128)[:, :rank].repeat(4, 1, 1)
            spare = _write_basis(
                tmp_path / f"basis_{rank}.safetensors",
                model_path,
                basis,
                method="identity" if rank == 128 else "state_pca",
            )
        with vllm_runner(
            model_path,
            revision=REVISION,
            dtype="bfloat16",
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=32,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            mamba_ssm_cache_dtype="float32",
            mamba_spare=spare,
            num_gpu_blocks_override=32,
            kv_cache_memory_bytes=32 * 1024**2,
            skip_tokenizer_init=True,
            enforce_eager=not graph,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY" if graph else "NONE",
                "cudagraph_capture_sizes": [1, 2] if graph else [],
            },
        ) as runner:
            (cache,) = runner.apply_model(_state_allocation)
            allocation.append(cache)
            prompts = [
                {"prompt_token_ids": [3, 4] * 23},
                {"prompt_token_ids": [5, 6, 7]},
                {"prompt_token_ids": [8, 9] * 16},
            ]
            params = SamplingParams(
                temperature=0, max_tokens=4, detokenize=False, ignore_eos=True
            )

            def generate(inputs, params=params):
                outputs = runner.get_llm().generate(inputs, params, use_tqdm=False)
                return [list(output.outputs[0].token_ids) for output in outputs]

            generated = generate(prompts)
            assert all(len(tokens) == 4 for tokens in generated)
            assert generate([prompts[-1], prompts[0]]) == [generated[-1], generated[0]]
            generations.append(generated)

    assert generations[0] == generations[1]
    assert allocation[0][:2] == allocation[1][:2]
    assert allocation[0][0] == allocation[2][0]
    assert allocation[0][1][-1] == 128 and allocation[2][1][-1] == 32
    assert allocation[2][2] < allocation[0][2]
