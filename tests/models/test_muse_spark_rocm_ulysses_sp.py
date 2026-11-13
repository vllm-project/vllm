# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm Ulysses prefill: option parsing, the all-to-all layouts on a
`TokenShard` (gloo, CPU) and, on gfx950 at TEP2, a random-weight model against
the token-sharded stream without Ulysses."""

import json
import zlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import (
    CUDAGraphMode,
    DeviceConfig,
    LoRAConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed import get_tp_group, parallel_state
from vllm.models.muse_spark.amd import ulysses_sp as usp
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..utils import multi_gpu_test
from .test_muse_spark_rocm_mono import TINY_TEXT, _checkpoint_config

ON = {"ulysses_prefill": True}


def _config(additional_config: dict, max_num_batched_tokens: int = 8192) -> VllmConfig:
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=4, distributed_executor_backend="mp"
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=max_num_batched_tokens,
            max_num_batched_tokens=max_num_batched_tokens,
            is_encoder_decoder=False,
        ),
        additional_config=additional_config,
    )


def _all_steps_captured() -> VllmConfig:
    config = _config(ON)
    compilation = config.compilation_config
    compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    compilation.max_cudagraph_capture_size = 8192
    return config


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "options, match",
    [
        ({"ulysses_prefil": True}, "Unknown additional_config key"),
        ({"ulysses_prefill": "false"}, "JSON boolean"),
    ],
)
def test_ulysses_prefill_rejects_invalid_options(options: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        usp.ulysses_prefill_enabled(_config(options), 1024)


@pytest.mark.cpu_test
def test_ulysses_prefill_refuses_lora() -> None:
    config = _config(ON)
    config.lora_config = LoRAConfig()
    with pytest.raises(ValueError, match="LoRA"):
        usp.ulysses_prefill_enabled(config, 1024)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "config, token_shard_min_tokens",
    [
        pytest.param(_config({}), 1024, id="off"),
        pytest.param(_config(ON), 0, id="stream-never-sharded"),
        pytest.param(_config(ON, max_num_batched_tokens=512), 1024, id="above-mnbt"),
        pytest.param(_all_steps_captured(), 1024, id="all-steps-captured"),
    ],
)
def test_ulysses_prefill_is_off_where_no_step_can_take_it(
    config: VllmConfig, token_shard_min_tokens: int
) -> None:
    assert not usp.ulysses_prefill_enabled(config, token_shard_min_tokens)


def _layout_worker(rank: int, world_size: int, port: int, num_tokens: int) -> None:
    cols = 6
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    # A real TP group, minus the device communicator so it runs on CPU.
    parallel_state._TP = parallel_state.GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="tp",
    )
    from vllm.models.muse_spark.amd.model import TokenShard

    class _GlooTokenShard(TokenShard):
        """`TokenShard` whose gather runs on gloo (the custom op is GPU-only)."""

        def gather(self, x: torch.Tensor) -> torch.Tensor:
            out = x.new_empty((self.rows * self.world, *x.shape[1:]))
            dist.all_gather_into_tensor(
                out, x.contiguous(), group=get_tp_group().device_group
            )
            return out[: self.num_tokens]

    shard = _GlooTokenShard(num_tokens)
    shard_len = shard.rows
    padded_len = shard_len * world_size

    def global_tensor(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(shape, generator=gen).to(torch.bfloat16)

    # [rows, W*cols] token shards -> all tokens of this rank's columns.
    padded = torch.zeros(padded_len, world_size * cols, dtype=torch.bfloat16)
    padded[:num_tokens] = global_tensor(1, (num_tokens, world_size * cols))
    mine = shard.local(padded[:num_tokens]).contiguous()
    got = usp.a2a_tokens_to_heads(mine, cols, shard)
    assert torch.equal(got, padded[:num_tokens, rank * cols : (rank + 1) * cols])

    # [T, cols] of this rank's heads -> [rows, W*cols] of this rank's tokens.
    heads = [global_tensor(10 + r, (num_tokens, cols)) for r in range(world_size)]
    got = usp.a2a_heads_to_tokens(heads[rank], shard)
    ref = torch.zeros(padded_len, world_size * cols, dtype=torch.bfloat16)
    for r in range(world_size):
        ref[:num_tokens, r * cols : (r + 1) * cols] = heads[r]
    assert torch.equal(got, ref[rank * shard_len : (rank + 1) * shard_len])

    # Byte-typed gathers (the MXFP4 expert input) round-trip exactly.
    raw = torch.randint(0, 256, (shard_len, 16), dtype=torch.uint8)
    gathered = usp.gather_bytes(raw, shard)
    ref_bytes = [torch.empty_like(raw) for _ in range(world_size)]
    dist.all_gather(ref_bytes, raw, group=get_tp_group().device_group)
    assert torch.equal(gathered, torch.cat(ref_bytes)[:num_tokens])
    dist.destroy_process_group()


@pytest.mark.cpu_test
@pytest.mark.parametrize("world_size, num_tokens", [(2, 8), (3, 11), (4, 13)])
def test_token_shard_layouts_match_the_tp_layout(
    world_size: int, num_tokens: int
) -> None:
    """Both Ulysses all-to-all exchanges and the byte gather on a `TokenShard`,
    including a token count that is not a multiple of the TP size (padded shards)."""
    mp.spawn(
        _layout_worker,
        args=(world_size, get_open_port(), num_tokens),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens, k", [(1, 8), (5, 8), (7, 3)])
def test_expand_rows_repeats_each_row(tokens: int, k: int) -> None:
    for dtype, width in ((torch.bfloat16, 8), (torch.uint8, 32), (torch.uint8, 6)):
        x = torch.randint(0, 100, (tokens, width)).to(dtype)
        out = usp.expand_rows(x, k)
        assert out.is_contiguous()
        assert torch.equal(out, x.repeat_interleave(k, dim=0))


def _random_weights(model: torch.nn.Module) -> int:
    """Replace the BF16/FP32 weights with O(1) random ones (matrices scaled by
    1/sqrt(fan-in), vectors in [0.5, 1.5]; distinct per TP rank where TP-sharded),
    rebuild the Ulysses copies and return how many attention layers have them."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    sharded_types = (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding)
    rank = get_tensor_model_parallel_rank()
    for prefix, module in model.named_modules():
        tag = rank if isinstance(module, sharded_types) else -1
        for name, param in module.named_parameters(recurse=False):
            if param.dtype not in (torch.bfloat16, torch.float32) or param.dim() == 0:
                continue
            seed = zlib.crc32(f"{prefix}.{name}:{tag}".encode())
            gen = torch.Generator(device=param.device).manual_seed(seed)
            if param.dim() == 1:
                param.data.uniform_(0.5, 1.5, generator=gen)
            else:
                param.data.normal_(generator=gen).mul_(param.shape[-1] ** -0.5)
    layers = [m for m in model.modules() if getattr(m, "ulysses_weights", None)]
    for layer in layers:
        layer.build_ulysses_weights()
    return len(layers)


LAYERS = 2
# One step of an odd token count: padded shards.
PROMPT = [t % 497 + 3 for t in range(2561)]


def _tiny_llm(tmp_path, monkeypatch, additional_config: dict):
    """A two-layer Muse Spark at TEP2 with O(1) random weights (`_random_weights`).
    Skips unless gfx950."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    # `apply_model` and `collective_rpc` send functions to the TP workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("Muse Spark on ROCm needs AITER's gfx950 kernels")
    text = TINY_TEXT | {
        "num_hidden_layers": LAYERS,
        **{
            key: TINY_TEXT[key][:LAYERS]
            for key in ("layer_types", "layer_rope_theta", "mlp_layer_types")
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(_checkpoint_config(text)))
    llm = LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        language_model_only=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=4096,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        additional_config=additional_config,
    )
    built = llm.apply_model(_random_weights)
    assert built == [LAYERS if additional_config else 0] * 2
    return llm


def _prompt_logprobs(llm) -> torch.Tensor:
    from vllm import SamplingParams

    out = llm.generate(
        [{"prompt_token_ids": PROMPT}],
        SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0),
    )[0]
    values = [
        lp[tok].logprob
        for tok, lp in zip(PROMPT[1:], out.prompt_logprobs[1:], strict=True)
    ]
    return torch.tensor(values)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark Ulysses needs ROCm"
)
@multi_gpu_test(num_gpus=2)
def test_ulysses_prefill_matches_the_token_sharded_stream(
    tmp_path, monkeypatch
) -> None:
    """Prompt logprobs of one Ulysses step stay within the FP8 projection error of
    the token-sharded stream without Ulysses. A wrong head or rank order moves them
    by far more."""
    llm = _tiny_llm(tmp_path, monkeypatch, {})
    reference = _prompt_logprobs(llm)
    del llm
    llm = _tiny_llm(tmp_path, monkeypatch, ON)
    ulysses = _prompt_logprobs(llm)
    delta = (reference - ulysses).abs()
    assert delta.max() > 0, "the Ulysses path did not run"
    assert delta.mean() < 0.06, delta.mean()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _parts(w: "usp.UlyssesWeights") -> dict[str, list[torch.Tensor]]:
    return {part: [_bits(t) for t in getattr(w, part)] for part in ("qkv", "gate", "o")}


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _reload_attention(worker, exponent: int) -> dict:
    """Reload every attention QKV and O weight (kernel format) with its columns
    scaled by 2**exponent and 2**-exponent alternately, which is exact in BF16 and
    changes the per-row FP8 payload, not only its scales. Returns the
    addresses of the Ulysses copies, the copies (qkv, gate, o) that differ from
    ones freshly built from the reloaded weights, and those the reload left
    unchanged."""
    model = worker.get_model()
    layers = [m for m in model.modules() if getattr(m, "ulysses_weights", None)]
    before = [
        {k: [t.clone() for t in v] for k, v in _parts(m.ulysses_weights).items()}
        for m in layers
    ]
    weights = []
    for name, param in model.named_parameters():
        if name.endswith(("qkv_proj.weight", "o_proj.weight")):
            cols = torch.arange(param.shape[1], device=param.device)
            scale = torch.exp2(exponent * (1 - 2 * (cols % 2)).float())
            weights.append((name, param * scale.to(param.dtype)))
    worker.reload_weights(weights_iterator=iter(weights), is_checkpoint_format=False)
    stale, unchanged = [], []
    for i, (m, old) in enumerate(zip(layers, before, strict=True)):
        held = _parts(m.ulysses_weights)
        fresh = _parts(usp.build_weights(m.qkv_proj, m.o_proj, m.q_size, m.kv_size))
        stale += [f"{i}.{k}" for k in held if not _same(held[k], fresh[k])]
        unchanged += [f"{i}.{k}" for k in held if _same(held[k], old[k])]
    return {
        "ptrs": [t.data_ptr() for m in layers for t in m.ulysses_weights.tensors],
        "stale": stale,
        "unchanged": unchanged,
    }


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark Ulysses needs ROCm"
)
@multi_gpu_test(num_gpus=2)
def test_ulysses_prefill_follows_weight_reloads(tmp_path, monkeypatch) -> None:
    """A weight reload refills each Ulysses copy (qkv, gate, o) in place: every
    held copy equals one built fresh from the reloaded weights, and the Ulysses
    step changes with perturbed weights and returns to its logprobs with the
    original ones."""
    llm = _tiny_llm(tmp_path, monkeypatch, ON)

    def reload(exponent: int) -> list[int]:
        result = llm.collective_rpc(_reload_attention, args=(exponent,))
        assert [r["stale"] for r in result] == [[], []]
        if exponent:
            assert [r["unchanged"] for r in result] == [[], []]
        return [r["ptrs"] for r in result]

    ptrs = reload(0)
    original = _prompt_logprobs(llm)
    assert reload(1) == ptrs
    perturbed = _prompt_logprobs(llm)
    assert reload(-1) == ptrs
    restored = _prompt_logprobs(llm)
    assert (perturbed - original).abs().mean() > 0.1
    torch.testing.assert_close(restored, original, rtol=0, atol=1e-3)
