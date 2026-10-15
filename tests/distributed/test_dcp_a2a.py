# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP A2A communication backend (no GPU required).

Tests cover:
1. DCP A2A config validation (--dcp-comm-backend)
2. KVP group function exists
3. LSE-weighted combination correctness
"""

import math

import multiprocess as mp
import pytest
import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config.parallel import ParallelConfig
from vllm.utils.network_utils import get_file_store_init_method
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


class _FakeCPGroup:
    def __init__(self, world_size: int, device_group: dist.ProcessGroup):
        self.world_size = world_size
        self.device_group = device_group


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _packed_a2a_reference(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    is_lse_base_on_e: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.ops.dcp import _lse_weighted_combine

    B, _H, D = cp_attn_out.shape
    outputs = (
        cp_attn_out.view(B, world_size, h_per_rank, D)
        .permute(1, 0, 2, 3)
        .contiguous()
        .float()
    )
    lses = cp_attn_lse.view(B, world_size, h_per_rank).permute(1, 0, 2).contiguous()
    return _lse_weighted_combine(
        outputs,
        lses,
        return_lse=True,
        is_lse_base_on_e=is_lse_base_on_e,
    )


def _assert_packed_a2a_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=3e-2, atol=3e-2
        )


def _distributed_run(fn, world_size: int, extra_env: dict[str, str]) -> None:
    distributed_init_method = get_file_store_init_method()
    processes: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "DISTRIBUTED_INIT_METHOD": distributed_init_method,
            **extra_env,
        }
        process = mp.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join(timeout=120)

    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0


class TestDCPCommBackendConfig:
    """Test --dcp-comm-backend config validation."""

    def test_default_is_ag_rs(self):
        """Comm backend resolves to ag_rs unless the model asks otherwise."""
        config = ParallelConfig()
        config.set_dcp_defaults()
        assert config.dcp_comm_backend == "ag_rs"

    def test_a2a_with_dcp_valid(self):
        """A2A backend is valid when DCP > 1."""
        config = ParallelConfig(
            dcp_comm_backend="a2a",
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
        )
        assert config.dcp_comm_backend == "a2a"

    def test_invalid_backend_rejected(self):
        """Invalid backend values are rejected."""
        with pytest.raises(ValueError, match="must be one of|Input should be"):
            ParallelConfig(
                dcp_comm_backend="invalid",
            )

    def test_ag_rs_with_dcp_1_valid(self):
        """ag_rs backend is valid with DCP=1 (no DCP)."""
        config = ParallelConfig(
            dcp_comm_backend="ag_rs",
            decode_context_parallel_size=1,
        )
        assert config.dcp_comm_backend == "ag_rs"

    @pytest.fixture
    def fi_config(self, monkeypatch):
        monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_Q_GATHER", "0")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_KV_GATHER", "0")
        monkeypatch.setenv("VLLM_USE_DIRECT_DCP_A2A", "0")
        return dict(
            dcp_comm_backend="fi_a2a_fused",
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
            disable_custom_all_reduce=True,
        )

    def test_fi_opt_in_preserves_model_defaults(self, fi_config):
        config = ParallelConfig(**fi_config)
        config.set_dcp_defaults(comm_backend="a2a")
        assert config.dcp_comm_backend == "fi_a2a_fused"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"decode_context_parallel_size": 1},
            {"prefill_context_parallel_size": 4},
            {"disable_custom_all_reduce": False},
        ],
    )
    def test_fi_rejects_unsupported_parallelism(self, fi_config, overrides):
        with pytest.raises(ValueError, match="fi_a2a_fused requires"):
            ParallelConfig(**(fi_config | overrides))

    @pytest.mark.parametrize(
        "flag",
        [
            "VLLM_ALLREDUCE_USE_SYMM_MEM",
            "VLLM_USE_DIRECT_DCP_Q_GATHER",
            "VLLM_USE_DIRECT_DCP_KV_GATHER",
            "VLLM_USE_DIRECT_DCP_A2A",
        ],
    )
    def test_fi_rejects_conflicting_collectives(self, fi_config, monkeypatch, flag):
        monkeypatch.setenv(flag, "1")
        with pytest.raises(ValueError, match="fi_a2a_fused requires"):
            ParallelConfig(**fi_config)

    @pytest.mark.parametrize(
        "hf_config,error",
        [
            (
                '{"model_type":"llama","architectures":["LlamaForCausalLM"],'
                '"num_attention_heads":16,"max_position_embeddings":128}',
                "greater than the model's total number of KV heads",
            ),
            (
                '{"model_type":"kimi_k3",'
                '"architectures":["KimiK3ForConditionalGeneration"],'
                '"text_config":{"num_attention_heads":16,"kv_lora_rank":64,'
                '"qk_rope_head_dim":32,"qk_nope_head_dim":128,"v_head_dim":128,'
                '"max_position_embeddings":128}}',
                "model-owned symmetric-memory",
            ),
        ],
        ids=["mha", "dedicated-mla"],
    )
    def test_fi_rejects_unsupported_models(self, fi_config, tmp_path, hf_config, error):
        from vllm.config.model import ModelConfig

        (tmp_path / "config.json").write_text(hf_config)
        model = ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        )
        with pytest.raises(ValueError, match=error):
            model.verify_with_parallel_config(ParallelConfig(**fi_config))

    @pytest.mark.parametrize(
        "num_kv_heads,dcp_size", [(1, 4), (2, 2)], ids=["mqa", "gqa"]
    )
    def test_fi_accepts_gqa_mqa_models(
        self, fi_config, tmp_path, num_kv_heads, dcp_size
    ):
        from vllm.config.model import ModelConfig

        (tmp_path / "config.json").write_text(
            '{"model_type":"llama","architectures":["LlamaForCausalLM"],'
            '"num_attention_heads":16,"max_position_embeddings":128,'
            f'"num_key_value_heads":{num_kv_heads}' + "}"
        )
        model = ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        )
        parallel = ParallelConfig(
            **(fi_config | {"decode_context_parallel_size": dcp_size})
        )
        model.verify_with_parallel_config(parallel)

    def test_fi_rejects_non_flashinfer_attention_backend(self, fi_config):
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.attention import Attention
        from vllm.utils.torch_utils import set_default_torch_dtype
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        config = VllmConfig(parallel_config=ParallelConfig(**fi_config))
        with (
            set_default_torch_dtype(torch.bfloat16),
            set_current_vllm_config(config),
            pytest.raises(ValueError, match="fi_a2a_fused.*FLASHINFER"),
        ):
            Attention(
                num_heads=4,
                head_size=128,
                scale=128**-0.5,
                num_kv_heads=1,
                attn_backend=FlashAttentionBackend,
            )


class TestLSEWeightedCombine:
    """Test LSE-weighted combination logic (CPU only, no GPU).

    The _lse_weighted_combine function is the reference implementation
    that verifies the Triton kernel's correctness. It computes:

        result[b,h,d] = sum_n(w_n * output_n[b,h,d])

    where w_n = softmax(lse_n) = exp(lse_n) / sum_k(exp(lse_k))
    """

    def test_importable(self):
        """Verify _lse_weighted_combine is importable."""
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        assert callable(_lse_weighted_combine)

    def test_single_rank(self):
        """Single rank: output unchanged."""
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        # N=1, B=2, H=4, D=8
        outputs = torch.randn(1, 2, 4, 8)
        lses = torch.randn(1, 2, 4)

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (2, 4, 8)
        torch.testing.assert_close(result, outputs.squeeze(0), rtol=1e-5, atol=1e-5)

    def test_equal_lse(self):
        """Equal LSE values: outputs averaged equally."""
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        _N, B, H, D = 2, 1, 1, 4
        outputs = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],  # Rank 0
                [[[5.0, 6.0, 7.0, 8.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[0.0]],  # Rank 0
                [[0.0]],  # Rank 1
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        expected = (outputs[0] + outputs[1]) / 2
        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_dominant_rank(self):
        """Different LSE values: larger LSE gets more weight."""
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[0.0, 0.0]]],  # Rank 0
                [[[1.0, 1.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[-100.0]],  # Rank 0: negligible contribution
                [[0.0]],  # Rank 1: dominant
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, outputs[1], atol=1e-5, rtol=1e-5)

    def test_empty_shard_ignores_undefined_output(self):
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        outputs = torch.tensor([[[[float("nan")]]], [[[3.0]]]])
        lses = torch.tensor([[[-float("inf")]], [[0.0]]])

        result = _lse_weighted_combine(outputs, lses)

        torch.testing.assert_close(result, outputs[1])

    def test_ag_rs_masks_empty_shard_and_padded_lse(self, monkeypatch):
        import vllm.v1.attention.ops.dcp as dcp

        class FakeGroup:
            world_size = 2
            rank_in_group = 0

            def all_gather(self, tensor, dim):
                assert dim == 0
                return torch.cat((tensor, tensor), dim=dim)

        monkeypatch.setattr(
            dcp,
            "correct_attn_out",
            lambda output, lses, *args, **kwargs: (output, lses[0]),
        )
        output = torch.ones(7, 1, 1)
        lse = torch.ones(7, 1)
        seq_lens = torch.tensor([0, 2], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 5], dtype=torch.int32)

        _, masked_lse = dcp._cp_lse_common(
            output,
            lse,
            FakeGroup(),
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
        )

        assert torch.isneginf(masked_lse[:1]).all()
        torch.testing.assert_close(masked_lse[1:5], torch.ones_like(masked_lse[1:5]))
        assert torch.isneginf(masked_lse[5:]).all()

    def test_mathematically_correct(self):
        """Verify mathematical correctness of LSE combination."""
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        outputs = torch.tensor(
            [
                [[[2.0, 4.0]]],
                [[[6.0, 8.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],  # exp(1) ≈ 2.718
                [[2.0]],  # exp(2) ≈ 7.389
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        w0 = math.exp(1) / (math.exp(1) + math.exp(2))
        w1 = math.exp(2) / (math.exp(1) + math.exp(2))
        expected = torch.tensor([[[w0 * 2.0 + w1 * 6.0, w0 * 4.0 + w1 * 8.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_return_lse(self):
        """return_lse=True returns global LSE (logsumexp of inputs)."""
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(outputs, lses, return_lse=True)

        expected_global_lse = math.log(math.exp(1) + math.exp(2))

        assert result.shape == (B, H, D)
        assert global_lse.shape == (B, H)
        assert abs(global_lse.item() - expected_global_lse) < 1e-5

    def test_base2_return_lse(self):
        """Base-2 LSE mode returns log2-sum-exp2 global LSE."""
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(
            outputs,
            lses,
            return_lse=True,
            is_lse_base_on_e=False,
        )

        expected_global_lse = math.log2(2**1 + 2**2)
        w0 = 2**1 / (2**1 + 2**2)
        w1 = 2**2 / (2**1 + 2**2)
        expected = torch.tensor([[[w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            global_lse,
            torch.tensor([[expected_global_lse]]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_lse_pack_dim(self):
        """Packed A2A stores one fp32 LSE in output-dtype lanes."""
        from vllm.v1.attention.ops.dcp import _dcp_a2a_lse_pack_dim

        assert _dcp_a2a_lse_pack_dim(torch.bfloat16) == 2
        assert _dcp_a2a_lse_pack_dim(torch.float16) == 2
        assert _dcp_a2a_lse_pack_dim(torch.float32) == 1


class TestPackedA2AKernels:
    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
    @pytest.mark.parametrize("return_lse", [False, True])
    @pytest.mark.parametrize("is_lse_base_on_e", [False, True])
    def test_pack_unpack_combine_matches_reference(
        self,
        dtype_name: str,
        return_lse: bool,
        is_lse_base_on_e: bool,
    ):
        from vllm.v1.attention.ops.dcp import (
            _dcp_a2a_lse_pack_dim,
            _dcp_a2a_pack_send,
            _dcp_a2a_unpack_combine,
        )

        torch.manual_seed(0)
        dtype = _dtype_from_name(dtype_name)
        device = torch.device("cuda")
        world_size, B, h_per_rank, D = 4, 7, 2, 32
        H = world_size * h_per_rank
        cp_attn_out = torch.randn(B, H, D, device=device, dtype=dtype)
        cp_attn_lse = torch.randn(B, H, device=device, dtype=dtype)
        lse_pack_dim = _dcp_a2a_lse_pack_dim(dtype)
        send_buffer = torch.empty(
            (world_size, B, h_per_rank, D + lse_pack_dim),
            device=device,
            dtype=dtype,
        )

        _dcp_a2a_pack_send(
            cp_attn_out,
            cp_attn_lse,
            send_buffer,
            world_size,
            h_per_rank,
            D,
            lse_pack_dim,
        )
        actual = _dcp_a2a_unpack_combine(
            send_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
        )
        expected_out, expected_lse = _packed_a2a_reference(
            cp_attn_out, cp_attn_lse, world_size, h_per_rank, is_lse_base_on_e
        )

        if return_lse:
            actual_out, actual_lse = actual
            _assert_packed_a2a_close(actual_out, expected_out, dtype)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)
        else:
            _assert_packed_a2a_close(actual, expected_out, dtype)

    @pytest.mark.skipif(
        torch.accelerator.device_count() < 1, reason="CUDA is required."
    )
    def test_empty_seq_lens_ignore_undefined_output(self):
        from vllm.v1.attention.ops.dcp import (
            _dcp_a2a_lse_pack_dim,
            _dcp_a2a_pack_send,
            _dcp_a2a_unpack_combine,
        )

        device = torch.device("cuda")
        world_size, num_tokens, h_per_rank, head_dim = 2, 5, 1, 32
        num_heads = world_size * h_per_rank
        output = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        output[:1] = float("nan")
        lse = torch.randn(num_tokens, num_heads, device=device, dtype=output.dtype)
        seq_lens = torch.tensor([0, 2], device=device, dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 5], device=device, dtype=torch.int32)
        lse_pack_dim = _dcp_a2a_lse_pack_dim(output.dtype)
        send_buffer = torch.empty(
            (
                world_size,
                num_tokens,
                h_per_rank,
                head_dim + lse_pack_dim,
            ),
            device=device,
            dtype=output.dtype,
        )

        _dcp_a2a_pack_send(
            output,
            lse,
            send_buffer,
            world_size,
            h_per_rank,
            head_dim,
            lse_pack_dim,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
        )
        actual_output, actual_lse = _dcp_a2a_unpack_combine(
            send_buffer,
            head_dim,
            lse_pack_dim,
            return_lse=True,
            is_lse_base_on_e=True,
        )

        torch.testing.assert_close(
            actual_output[:1], torch.zeros_like(actual_output[:1])
        )
        assert torch.isneginf(actual_lse[:1]).all()
        assert torch.isfinite(actual_output[1:]).all()


def _distributed_packed_a2a_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    world_size = int(env["WORLD_SIZE"])
    torch.accelerator.set_device_index(local_rank)
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            init_method=env["DISTRIBUTED_INIT_METHOD"],
            rank=local_rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(
            backend="nccl",
            init_method=env["DISTRIBUTED_INIT_METHOD"],
            rank=local_rank,
            world_size=world_size,
        )
    use_workspace = env.get("USE_WORKSPACE") == "1"
    if use_workspace:
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(torch.device(f"cuda:{local_rank}"))
    try:
        from vllm.v1.attention.ops.dcp import dcp_a2a_lse_reduce

        dtype = _dtype_from_name(env["TEST_DTYPE"])
        return_lse = env["RETURN_LSE"] == "1"
        is_lse_base_on_e = env["LSE_BASE_E"] == "1"
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        B, h_per_rank, D = 5, 2, 32
        H = world_size * h_per_rank

        generator = torch.Generator(device=f"cuda:{local_rank}")
        generator.manual_seed(1234 + rank)
        cp_attn_out = torch.randn(
            B,
            H,
            D,
            device=f"cuda:{local_rank}",
            dtype=dtype,
            generator=generator,
        )
        cp_attn_lse = torch.randn(
            B,
            H,
            device=f"cuda:{local_rank}",
            dtype=dtype,
            generator=generator,
        )
        actual = dcp_a2a_lse_reduce(
            cp_attn_out,
            cp_attn_lse,
            _FakeCPGroup(world_size, dist.group.WORLD),
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
        )

        gathered_out = [torch.empty_like(cp_attn_out) for _ in range(world_size)]
        gathered_lse = [torch.empty_like(cp_attn_lse) for _ in range(world_size)]
        dist.all_gather(gathered_out, cp_attn_out)
        dist.all_gather(gathered_lse, cp_attn_lse)
        outputs = torch.stack(
            [
                t[:, rank * h_per_rank : (rank + 1) * h_per_rank, :]
                for t in gathered_out
            ],
            dim=0,
        ).float()
        lses = torch.stack(
            [t[:, rank * h_per_rank : (rank + 1) * h_per_rank] for t in gathered_lse],
            dim=0,
        )
        from vllm.v1.attention.ops.dcp import _lse_weighted_combine

        expected_out, expected_lse = _lse_weighted_combine(
            outputs,
            lses,
            return_lse=True,
            is_lse_base_on_e=is_lse_base_on_e,
        )

        if return_lse:
            actual_out, actual_lse = actual
            _assert_packed_a2a_close(actual_out, expected_out, dtype)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)
        else:
            _assert_packed_a2a_close(actual, expected_out, dtype)
    finally:
        if use_workspace:
            from vllm.v1.worker.workspace import reset_workspace_manager

            reset_workspace_manager()
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
)
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_distributed_packed_a2a_matches_reference(dtype_name: str):
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=4,
        extra_env={
            "TEST_DTYPE": dtype_name,
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
        },
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
)
def test_distributed_packed_a2a_with_workspace_matches_reference():
    _distributed_run(
        _distributed_packed_a2a_worker,
        world_size=4,
        extra_env={
            "TEST_DTYPE": "bfloat16",
            "RETURN_LSE": "1",
            "LSE_BASE_E": "1",
            "USE_WORKSPACE": "1",
        },
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need a CUDA GPU")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fi_dcp_valid_token_mask_reuses_storage(index_dtype):
    from vllm.v1.attention.ops.metadata import build_dcp_valid_token_mask

    seq_lens = torch.tensor([3, 0, 0, 9], device="cuda", dtype=index_dtype)
    query_start_loc = torch.tensor([0, 2, 2, 5, 6], device="cuda", dtype=index_dtype)
    mask = torch.empty(8, device="cuda", dtype=torch.bool)
    other_mask = torch.empty_like(mask)
    mask_ptr = mask.data_ptr()
    assert mask_ptr != other_mask.data_ptr()

    build_dcp_valid_token_mask(seq_lens, query_start_loc, mask)
    expected = torch.tensor(
        [True, True, False, False, False, True, False, False], device="cuda"
    )
    torch.testing.assert_close(mask, expected)

    seq_lens.copy_(torch.tensor([0, 9, 5, 0], device="cuda", dtype=index_dtype))
    query_start_loc.copy_(
        torch.tensor([0, 1, 1, 2, 2], device="cuda", dtype=index_dtype)
    )
    build_dcp_valid_token_mask(seq_lens, query_start_loc, other_mask)
    torch.testing.assert_close(mask, expected)
    expected = torch.tensor(
        [False, True, False, False, False, False, False, False], device="cuda"
    )
    torch.testing.assert_close(other_mask, expected)
    build_dcp_valid_token_mask(seq_lens, query_start_loc, mask)
    assert mask.data_ptr() == mask_ptr
    torch.testing.assert_close(mask, expected)

    build_dcp_valid_token_mask(seq_lens[:0], query_start_loc[:1], mask)
    assert not mask.any()
    torch.testing.assert_close(other_mask, expected)


def _distributed_dcp_manager_worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    local_rank = int(env["LOCAL_RANK"])
    world_size = int(env["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=env["DISTRIBUTED_INIT_METHOD"],
        rank=local_rank,
        world_size=world_size,
    )
    group = None
    graph = None
    manager = None
    combine = None
    fi_workspace = None
    first_workspace = None
    try:
        import vllm.v1.attention.ops.dcp as dcp
        from vllm.config import SchedulerConfig, VllmConfig
        from vllm.distributed.parallel_state import GroupCoordinator

        backend = env["TEST_MANAGER_BACKEND"]
        group = GroupCoordinator(
            [list(range(world_size))],
            local_rank,
            "nccl",
            use_device_communicator=backend == "fi_a2a_fused",
        )
        dcp.get_dcp_group = lambda: group
        if backend == "fi_a2a_fused":
            import torch.distributed._symmetric_memory as symm_mem

            from vllm.v1.attention.ops.metadata import build_dcp_valid_token_mask

        dtype = _dtype_from_name(env["TEST_DTYPE"])
        base_e = env["LSE_BASE_E"] == "1"
        heads_per_rank = 2 if world_size == 8 else 6
        flashinfer_impl = env.get("TEST_IMPL") == "flashinfer"
        head_dim, max_num_tokens = (128 if flashinfer_impl else 512), 128
        total_heads = world_size * heads_per_rank
        config = VllmConfig(
            parallel_config=ParallelConfig(
                dcp_comm_backend=backend,
                tensor_parallel_size=world_size,
                decode_context_parallel_size=world_size,
                disable_custom_all_reduce=True,
            ),
            scheduler_config=SchedulerConfig(
                max_num_batched_tokens=max_num_tokens,
                max_num_seqs=max_num_tokens,
                max_model_len=max_num_tokens,
                is_encoder_decoder=False,
            ),
        )
        if flashinfer_impl:
            from vllm.config import set_current_vllm_config
            from vllm.distributed import parallel_state
            from vllm.utils.torch_utils import set_default_torch_dtype
            from vllm.v1.attention.backends.flashinfer import FlashInferImpl

            parallel_state._DCP = group
            with set_default_torch_dtype(dtype), set_current_vllm_config(config):
                manager = FlashInferImpl(
                    num_heads=heads_per_rank,
                    head_size=head_dim,
                    scale=head_dim**-0.5,
                    num_kv_heads=1,
                    alibi_slopes=None,
                    sliding_window=None,
                    kv_cache_dtype="auto",
                )
            combine = manager._combine_dcp
        else:
            manager = dcp.MLADCPManager(
                config,
                device,
                heads_per_rank,
                576,
                head_dim,
                dtype,
                dtype,
                None,
                base_e,
                False,
            )
            combine = manager.combine
        if backend == "fi_a2a_fused":
            fi_workspace = (
                manager._fi_dcp_workspace if flashinfer_impl else combine.func.__self__
            )
            stream_id = int(torch.cuda.current_stream(device).cuda_stream)
            first_workspace = fi_workspace._workspaces[stream_id][0]
            if not flashinfer_impl:
                query = (
                    torch.arange(heads_per_rank, device=device, dtype=dtype)
                    .add(local_rank * heads_per_rank)
                    .view(1, heads_per_rank, 1)
                    .expand(3, heads_per_rank, 576)
                    .contiguous()
                )
                gathered_query = manager.query_gather(query)
                expected_query = (
                    torch.arange(total_heads, device=device, dtype=dtype)
                    .view(1, total_heads, 1)
                    .expand(3, total_heads, 576)
                )
                torch.testing.assert_close(
                    gathered_query, expected_query, rtol=0, atol=0
                )
        generator = torch.Generator(device=device)
        generator.manual_seed(1234 + local_rank)
        head_slice = slice(
            local_rank * heads_per_rank, (local_rank + 1) * heads_per_rank
        )

        def reference(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
            output, lse = output.contiguous(), lse.contiguous()
            gathered_output = [torch.empty_like(output) for _ in range(world_size)]
            gathered_lse = [torch.empty_like(lse) for _ in range(world_size)]
            dist.all_gather(gathered_output, output)
            dist.all_gather(gathered_lse, lse)
            expected, _ = _packed_a2a_reference(
                torch.cat([value[:, head_slice] for value in gathered_output], dim=1),
                torch.cat([value[:, head_slice] for value in gathered_lse], dim=1),
                world_size,
                heads_per_rank,
                base_e,
            )
            return expected

        all_valid_mask = (
            torch.ones(max_num_tokens + 1, device=device, dtype=torch.bool)
            if flashinfer_impl
            else None
        )

        def all_valid_kwargs(num_tokens: int) -> dict[str, torch.Tensor]:
            if all_valid_mask is None:
                return {}
            return {"valid_token_mask": all_valid_mask[:num_tokens]}

        # Exercise destination/head layout, strided inputs, and maximum capacity.
        for num_tokens, strided in ((1, False), (17, True), (128, True)):
            storage_heads = 128 if strided else total_heads
            output = torch.randn(
                num_tokens,
                storage_heads,
                head_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )[:, :total_heads]
            if strided and backend == "fi_a2a_fused":
                lse = torch.randn(
                    storage_heads,
                    num_tokens,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                ).transpose(0, 1)[:, :total_heads]
            else:
                lse = torch.randn(
                    num_tokens,
                    storage_heads,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )[:, :total_heads]
            if strided:
                assert not output.is_contiguous()
                assert not lse.is_contiguous()
            actual = combine(output, lse, **all_valid_kwargs(num_tokens))
            torch.accelerator.synchronize()
            _assert_packed_a2a_close(actual, reference(output, lse), dtype)

        if backend == "fi_a2a_fused":
            oversized = torch.empty(
                max_num_tokens + 1, total_heads, head_dim, device=device, dtype=dtype
            )
            oversized_lse = torch.empty(
                oversized.shape[:2], device=device, dtype=torch.float32
            )
            with pytest.raises(ValueError, match="capacity"):
                combine(
                    oversized, oversized_lse, **all_valid_kwargs(max_num_tokens + 1)
                )
            with pytest.raises(ValueError, match="FP32"):
                combine(output[:1], lse[:1].half(), **all_valid_kwargs(1))

        # One globally empty request, one rank-local empty request per source,
        # variable query lengths, and implicit CUDA graph padding.
        # Native direct does not mask rows beyond query_start_loc[-1].
        num_padding = 0 if env["VLLM_USE_DIRECT_DCP_A2A"] == "1" else 3
        query_lens = torch.tensor(
            [1, 3, 2, *([1] * (world_size - 1))]
            + ([0, 1, 0] if backend == "fi_a2a_fused" else []),
            device=device,
            dtype=torch.int32,
        )
        seq_lens = torch.arange(
            3, 3 + len(query_lens), device=device, dtype=torch.int32
        )
        seq_lens[0] = seq_lens[1 + local_rank] = 0
        query_start_loc = torch.cat(
            (query_lens.new_zeros(1), query_lens.cumsum(0, dtype=torch.int32))
        )
        empty_rows = torch.cat(
            (
                torch.repeat_interleave(seq_lens == 0, query_lens),
                torch.ones(num_padding, device=device, dtype=torch.bool),
            )
        )
        num_tokens = empty_rows.numel()
        empty_output = torch.randn(
            num_tokens,
            total_heads,
            head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        empty_lse = torch.randn(
            num_tokens,
            total_heads,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        empty_output[empty_rows] = float("nan")
        empty_lse[empty_rows] = 0.0
        empty_kwargs = {}
        if backend == "fi_a2a_fused":
            valid_token_mask = torch.empty(
                max_num_tokens, device=device, dtype=torch.bool
            )
            build_dcp_valid_token_mask(seq_lens, query_start_loc, valid_token_mask)
            torch.testing.assert_close(valid_token_mask[:num_tokens], ~empty_rows)
            missing_mask_error = (
                "metadata token mask" if flashinfer_impl else "valid_token_mask"
            )
            with pytest.raises(ValueError, match=missing_mask_error):
                combine(
                    empty_output,
                    empty_lse,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
            empty_kwargs["valid_token_mask"] = valid_token_mask[:num_tokens]
        actual = combine(
            empty_output,
            empty_lse,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            **empty_kwargs,
        )
        torch.accelerator.synchronize()
        empty_output[empty_rows] = 0
        empty_lse[empty_rows] = float("-inf")
        _assert_packed_a2a_close(actual, reference(empty_output, empty_lse), dtype)
        assert torch.equal(actual[:1], torch.zeros_like(actual[:1]))
        if num_padding:
            assert torch.equal(
                actual[-num_padding:], torch.zeros_like(actual[-num_padding:])
            )

        if backend == "fi_a2a_fused":
            # All local shards empty, including no requests during graph warmup.
            empty_output.fill_(float("nan"))
            empty_lse.zero_()
            for lengths, boundaries in (
                (torch.zeros_like(seq_lens), query_start_loc),
                (seq_lens[:0], query_start_loc[:1]),
            ):
                build_dcp_valid_token_mask(lengths, boundaries, valid_token_mask)
                actual = combine(
                    empty_output,
                    empty_lse,
                    seq_lens=lengths,
                    query_start_loc=boundaries,
                    valid_token_mask=valid_token_mask[:num_tokens],
                )
                torch.accelerator.synchronize()
                assert torch.equal(actual, torch.zeros_like(actual))

            nonfinite_output = output[:4].clone()
            nonfinite_output[:3] = float("nan")
            nonfinite_lse = (
                torch.tensor(
                    [float("nan"), float("inf"), float("-inf"), 0.0], device=device
                )
                .unsqueeze(1)
                .expand(4, total_heads)
                .contiguous()
            )
            actual = combine(nonfinite_output, nonfinite_lse, **all_valid_kwargs(4))
            torch.accelerator.synchronize()
            nonfinite_output[:3] = 0
            nonfinite_lse[:3] = float("-inf")
            _assert_packed_a2a_close(
                actual, reference(nonfinite_output, nonfinite_lse), dtype
            )
            assert torch.equal(actual[:3], torch.zeros_like(actual[:3]))

        # Warm a second stream, then change inputs between graph replays.
        graph_kwargs = {}
        if backend == "fi_a2a_fused":
            graph_seq_lens = torch.tensor([7, 5], device=device, dtype=torch.int64)
            graph_query_start_loc = torch.tensor(
                [0, 32, 96], device=device, dtype=torch.int64
            )
            graph_mask = torch.empty_like(valid_token_mask)
            assert graph_mask.data_ptr() != valid_token_mask.data_ptr()
            graph_mask_ptr = graph_mask.data_ptr()
            build_dcp_valid_token_mask(
                graph_seq_lens, graph_query_start_loc, graph_mask
            )
            graph_kwargs = dict(
                seq_lens=graph_seq_lens,
                query_start_loc=graph_query_start_loc,
                valid_token_mask=graph_mask,
            )
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            combine(output, lse, **graph_kwargs)
        capture_stream.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            actual = combine(output, lse, **graph_kwargs)
        for replay in range(3):
            output.fill_(local_rank + replay + 1)
            lse.fill_(local_rank * (-1.0 if replay % 2 == 0 else 1.0))
            if backend == "fi_a2a_fused":
                split = 32 >> replay
                graph_query_start_loc.copy_(
                    graph_query_start_loc.new_tensor([0, split, 2 * split])
                )
                graph_seq_lens.copy_(
                    graph_seq_lens.new_tensor(
                        [
                            0 if local_rank == replay % world_size else 7,
                            0 if replay == 2 else 5,
                        ]
                    )
                )
                build_dcp_valid_token_mask(
                    graph_seq_lens, graph_query_start_loc, graph_mask
                )
                assert graph_mask.data_ptr() == graph_mask_ptr
                graph_empty_rows = torch.cat(
                    (
                        torch.repeat_interleave(graph_seq_lens == 0, split),
                        torch.ones(
                            max_num_tokens - 2 * split, device=device, dtype=torch.bool
                        ),
                    )
                )
                torch.testing.assert_close(graph_mask, ~graph_empty_rows)
                output[graph_empty_rows] = float("nan")
            torch.accelerator.synchronize()
            graph.replay()
            torch.accelerator.synchronize()
            if backend == "fi_a2a_fused":
                output[graph_empty_rows] = 0
                lse[graph_empty_rows] = float("-inf")
            _assert_packed_a2a_close(actual, reference(output, lse), dtype)
        if backend == "fi_a2a_fused":
            assert fi_workspace is not None and first_workspace is not None
            assert symm_mem.get_backend(device) == "NCCL"
            assert fi_workspace._workspaces[stream_id][0] is first_workspace
            capture_id = int(capture_stream.cuda_stream)
            assert set(fi_workspace._workspaces) == {stream_id, capture_id}
            assert (
                fi_workspace._workspaces[capture_id][0].data_ptr()
                != first_workspace.data_ptr()
            )
    finally:
        if graph is not None:
            torch.accelerator.synchronize()
            graph.reset()
        del combine, manager, fi_workspace, first_workspace
        if env["TEST_MANAGER_BACKEND"] == "fi_a2a_fused":
            from vllm.v1.attention.ops.flashinfer_dcp import (
                get_flashinfer_dcp_workspace,
            )

            get_flashinfer_dcp_workspace.cache_clear()
        if group is not None:
            group.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2,
    reason="Need at least two CUDA GPUs in one NCCL LSA/NVLink domain",
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("base_e", [False, True])
@pytest.mark.parametrize(
    "backend,direct",
    [("a2a", "0"), ("a2a", "1"), ("fi_a2a_fused", "0")],
    ids=["packed", "native", "fi"],
)
def test_distributed_dcp_manager_matches_reference(
    world_size, dtype, base_e, backend, direct
):
    """Compare each real dispatcher in a process with its own symmetric allocator."""
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs")
    if backend == "fi_a2a_fused":
        pytest.importorskip("flashinfer.comm.dcp_lse_reduce")
    _distributed_run(
        _distributed_dcp_manager_worker,
        world_size,
        {
            "TEST_MANAGER_BACKEND": backend,
            "TEST_DTYPE": dtype,
            "LSE_BASE_E": str(int(base_e)),
            "VLLM_USE_DIRECT_DCP_A2A": direct,
            "VLLM_USE_DIRECT_DCP_Q_GATHER": "0",
            "VLLM_USE_DIRECT_DCP_KV_GATHER": "0",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        },
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2,
    reason="Need at least two CUDA GPUs in one NCCL LSA/NVLink domain",
)
@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_distributed_flashinfer_decode_combine_matches_reference(world_size, dtype):
    """Exercise the GQA/MQA decode dispatcher with the existing combine matrix."""
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs")
    pytest.importorskip("flashinfer.comm.dcp_lse_reduce")
    _distributed_run(
        _distributed_dcp_manager_worker,
        world_size,
        {
            "TEST_IMPL": "flashinfer",
            "TEST_MANAGER_BACKEND": "fi_a2a_fused",
            "TEST_DTYPE": dtype,
            "LSE_BASE_E": "0",
            "VLLM_USE_DIRECT_DCP_A2A": "0",
            "VLLM_USE_DIRECT_DCP_Q_GATHER": "0",
            "VLLM_USE_DIRECT_DCP_KV_GATHER": "0",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        },
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
