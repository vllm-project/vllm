# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from transformers import PreTrainedConfig

from tests.models.utils import check_logprobs_close, dummy_hf_overrides
from tests.utils import multi_gpu_marks
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def _small_hyv4_config(config):
    # Apply to both target and MTP draft while retaining MLA/indexer dimensions.
    config.update(
        {
            "hidden_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "q_lora_rank": 128,
            "intermediate_size": 512,
            "moe_intermediate_size": 256,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "enable_ihc": True,
            "index_n_heads": 32,
            "index_topk": 2048,
            "layer_types": ["deepseek_sparse_attention"] * 2,
            "indexer_types": ["full", "shared"],
            "num_nextn_predict_layers": 1,
            "vocab_size": 256,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
    )
    return config


def _num_captured_graphs(_worker):
    from vllm.compilation.counter import compilation_counter

    return compilation_counter.num_cudagraph_captured


def _graph_execution_state(worker):
    from vllm.compilation.counter import compilation_counter

    # `cudagraph_mode` is returned by name: the enum does not survive the
    # collective_rpc round trip, it arrives as its bare value.
    return (
        compilation_counter.num_models_seen,
        compilation_counter.num_cudagraph_captured,
        worker.vllm_config.compilation_config.mode.name,
        worker.vllm_config.compilation_config.cudagraph_mode.name,
    )


@pytest.mark.parametrize(
    "tp,enforce_eager,mtp,dtype,kv_cache_dtype",
    [
        pytest.param(1, True, False, "bfloat16", "auto", id="eager-tp1"),
        pytest.param(1, True, False, "float16", "auto", id="eager-fp16"),
        pytest.param(1, True, False, "bfloat16", "fp8", id="eager-fp8-kv"),
        pytest.param(
            2,
            False,
            False,
            "bfloat16",
            "auto",
            id="graphs-tp2",
            marks=multi_gpu_marks(num_gpus=2),
        ),
        pytest.param(1, True, True, "bfloat16", "auto", id="eager-mtp"),
    ],
)
def test_hyv4_dummy_generation(
    vllm_runner,
    enable_pickle,
    tp: int,
    enforce_eager: bool,
    mtp: bool,
    dtype: str,
    kv_cache_dtype: str,
) -> None:
    """Exercise prefill, decode, and batch consistency with dummy HY weights."""
    with vllm_runner(
        "tencent/Hy4-preview",
        skip_tokenizer_init=True,
        load_format="dummy",
        disable_log_stats=not mtp,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        hf_overrides=_small_hyv4_config,
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2}
        if mtp
        else None,
        seed=17,
        # Exercise supported decode graphs independently of default graph policy.
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE
            if enforce_eager
            else CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[] if enforce_eager else [1, 2, 4],
        ),
    ) as model:
        if not enforce_eager:
            assert model.collective_rpc(_num_captured_graphs) == [3] * tp
        prompts = [
            {"prompt_token_ids": [11, 12, 13, 14]},
            {"prompt_token_ids": list(range(20, 84))},
        ]
        params = SamplingParams(
            temperature=0, max_tokens=4, ignore_eos=True, logprobs=1
        )
        outputs = model.llm.generate(prompts, params)
        assert len(outputs) == len(prompts)
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0]
            assert len(completion.token_ids) == 4
            assert completion.cumulative_logprob is not None
            assert math.isfinite(completion.cumulative_logprob)
            single = model.llm.generate([prompt], params)[0].outputs[0]
            assert single.token_ids == completion.token_ids

        if mtp:
            assert any(
                isinstance(metric, Counter)
                and metric.name == "vllm:spec_decode_num_drafts"
                and metric.value > 0
                for metric in model.llm.get_metrics()
            ), "MTP drafter did not run"


def test_hyv4_defaults_to_breakable_graphs(
    vllm_runner, enable_pickle, monkeypatch
) -> None:
    """Keep prefill/decode graph capture when the backbone is not compiled."""
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    with vllm_runner(
        "tencent/Hy4-preview",
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        hf_overrides=_small_hyv4_config,
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enforce_eager=False,
        seed=17,
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4],
        ),
    ) as model:
        for models_seen, captured, compile_mode, graph_mode in model.collective_rpc(
            _graph_execution_state
        ):
            assert models_seen == 0, "backbone should not enter torch.compile"
            assert compile_mode == CompilationMode.NONE.name
            assert graph_mode == CUDAGraphMode.FULL_AND_PIECEWISE.name
            assert captured > 0

        outputs = model.llm.generate(
            [
                {"prompt_token_ids": [11, 12, 13, 14]},
                {"prompt_token_ids": list(range(20, 84))},
            ],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1),
        )
        assert len(outputs) == 2
        for output in outputs:
            completion = output.outputs[0]
            assert len(completion.token_ids) == 4
            assert completion.cumulative_logprob is not None
            assert math.isfinite(completion.cumulative_logprob)


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="HYV4 DFly and HY4 Flash on ROCm require gfx950"
)

# The recipe checkpoint; dummy weights only need its config and tokenizer.
HY4_FLASH_FP8 = "tencent/Hy4-flash-FP8"
_ENGRAM_PRIMES = [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049]


def _small_hyv4_dfly_config(config):
    _small_hyv4_config(config)
    # DFly reads the target layer's own indexer, so it cannot be shared.
    config.update(
        {
            "indexer_types": ["full", "full"],
            "num_nextn_predict_layers": 0,
            "dfly_num_hidden_layers": 1,
            "dfly_target_index_layer": [1],
            "dfly_target_kv_layers": [1],
            "dfly_block_size": 4,
            "dfly_mask_token_id": 9,
        }
    )
    return config


def _small_hyv4_flash_dfly_config(config):
    # Four layers keep a sparse DFly target layer, SWA layers and Engram layer 2.
    config = dummy_hf_overrides(
        config,
        model_arch="HYV4FlashVLForConditionalGeneration",
        num_hidden_layers=4,
    )
    config.get_text_config().update(
        {
            "engram_layer_head_vocab_sizes": {
                "2": [_ENGRAM_PRIMES, [p + 1000 for p in _ENGRAM_PRIMES]]
            }
        }
    )
    config.update(
        {
            "dfly_num_hidden_layers": 1,
            "dfly_target_index_layer": [1],
            "dfly_target_kv_layers": [1],
            "dfly_block_size": 4,
        }
    )
    return config


@dataclass(frozen=True)
class _E2ECase:
    model: str
    hf_overrides: Callable[[PreTrainedConfig], PreTrainedConfig]
    prompts: tuple[tuple[int, ...], ...]
    # Chosen-token logprob drift allowed while two greedy runs agree: ROCm
    # sparse attention is not bitwise reproducible (observed max 0.011 BF16).
    # FP8 adds run-to-run drift from the AITER block-scale FP8 GEMM, which is
    # not deterministic for small-M batches (observed max 0.59).
    logprob_atol: float
    engine_kwargs: dict

    def runner_kwargs(self, graphs: bool, dfly: bool, tp: int = 1) -> dict:
        return dict(
            load_format="dummy",
            dtype="bfloat16",
            hf_overrides=self.hf_overrides,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=not graphs,
            disable_log_stats=not dfly,
            tensor_parallel_size=tp,
            speculative_config={"method": "dfly"} if dfly else None,
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
                if graphs
                else CUDAGraphMode.NONE,
                cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32] if graphs else [],
            ),
            **self.engine_kwargs,
        )


# Several prompts end in the same token so the outputs depend on attention.
_E2E_CASES = {
    "hy4": _E2ECase(
        model="tencent/Hy4-preview",
        hf_overrides=_small_hyv4_dfly_config,
        prompts=(
            (7,),
            (11, 12, 13, 14),
            (200, 100, 14),
            (5, 6, 7, 8, 9, 10, 14),
            tuple(range(20, 84)),
            tuple((i * 37) % 250 + 3 for i in range(150)),
        ),
        logprob_atol=0.05,
        engine_kwargs=dict(
            skip_tokenizer_init=True,
            max_model_len=256,
            max_num_batched_tokens=128,
            max_num_seqs=8,
            kv_cache_memory_bytes=128 * 1024 * 1024,
            seed=17,
        ),
    ),
    "hy4-flash-fp8": _E2ECase(
        model=HY4_FLASH_FP8,
        hf_overrides=_small_hyv4_flash_dfly_config,
        prompts=(
            (1000,),
            tuple(range(1007, 1012)),
            (500, 600, 1014),
            (3000, 1014),
            tuple(range(1014, 1078)),
            tuple(range(1021, 1321)),
            tuple(range(1028, 1728)),
        ),
        logprob_atol=0.75,
        engine_kwargs=dict(
            max_model_len=1024,
            max_num_batched_tokens=512,
            max_num_seqs=8,
            kv_cache_memory_bytes=512 * 1024 * 1024,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=0,
        ),
    ),
}

_MAX_TOKENS = 12


def _context_dependent_head(model) -> int:
    """Make dummy-weight greedy decoding depend on context.

    Dummy init fills same-shaped tensors identically, so the untied LM head
    equals the embedding, and `1e-3` norm weights drown out every sublayer:
    the model would only copy its last input token. Unit norms and a random
    LM head (logit std ~8) let attention and MoE steer the next token.
    """
    generator = torch.Generator().manual_seed(0)
    updated = 0
    for name, param in model.named_parameters():
        if name.endswith("norm.weight"):
            param.data.fill_(1.0)
        elif name.endswith("lm_head.weight"):
            head = torch.randn(param.shape, generator=generator)
            param.data.copy_(head * 8 / param.shape[-1] ** 0.5)
        else:
            continue
        updated += 1
    return updated


def _reinit_head(model) -> None:
    assert all(n > 0 for n in model.llm.apply_model(_context_dependent_head))


def _greedy(model, case: _E2ECase):
    params = SamplingParams(
        temperature=0, max_tokens=_MAX_TOKENS, ignore_eos=True, logprobs=5
    )
    prompts = [{"prompt_token_ids": list(p)} for p in case.prompts]
    results = []
    for output in model.llm.generate(prompts, params, use_tqdm=False):
        completion = output.outputs[0]
        assert len(completion.token_ids) == _MAX_TOKENS
        assert completion.logprobs is not None
        logprobs = [
            {token: lp.logprob for token, lp in step.items()}
            for step in completion.logprobs
        ]
        assert all(math.isfinite(lp) for step in logprobs for lp in step.values())
        results.append((list(completion.token_ids), "", logprobs))
    return results


def _assert_greedy_close(
    expected, actual, name_0: str, name_1: str, atol: float
) -> None:
    check_logprobs_close(
        outputs_0_lst=expected,
        outputs_1_lst=actual,
        name_0=name_0,
        name_1=name_1,
        warn_on_mismatch=False,
    )
    for (tokens_0, _, logprobs_0), (tokens_1, _, logprobs_1) in zip(expected, actual):
        for step, (token_0, token_1) in enumerate(zip(tokens_0, tokens_1)):
            if token_0 != token_1:
                break
            assert logprobs_1[step][token_1] == pytest.approx(
                logprobs_0[step][token_0], abs=atol
            ), f"{name_1} drifted from {name_0} at step {step} of {tokens_0}"


def _num_drafts(llm) -> int:
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if isinstance(metric, Counter) and metric.name == "vllm:spec_decode_num_drafts"
    )


@pytest.fixture
def hyv4_recipe_env(monkeypatch, enable_pickle):
    """Run with the AITER settings the HY4 ROCm recipe requires."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA", "1")
    rocm_aiter_ops.refresh_env_variables()
    assert rocm_aiter_ops.is_triton_sparse_mla_enabled()
    yield
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture(scope="module")
def eager_target_outputs() -> dict[tuple[str, int], list]:
    """Per-module cache of eager target-only greedy outputs.

    Keyed by `(case_id, tp)`: dummy weights are generated per rank, so each TP
    size is a different model.
    """
    return {}


def _eager_target(
    vllm_runner, cache: dict[tuple[str, int], list], case_id: str, tp: int = 1
) -> list:
    if (case_id, tp) not in cache:
        case = _E2E_CASES[case_id]
        with vllm_runner(
            case.model, **case.runner_kwargs(graphs=False, dfly=False, tp=tp)
        ) as model:
            _reinit_head(model)
            cache[case_id, tp] = _greedy(model, case)
    return cache[case_id, tp]


# Flash-FP8 is TP1 only: at TP2 even two eager runs flip FP8 near-tie tokens.
_TP2 = multi_gpu_marks(num_gpus=2)
_DFLY_PARAMS = [
    pytest.param(case_id, graphs, 1, id=f"{graphs_id}-{case_id}")
    for case_id in _E2E_CASES
    for graphs, graphs_id in ((False, "eager"), (True, "full-decode-graphs"))
] + [pytest.param("hy4", True, 2, id="tp2-full-decode-graphs-hy4", marks=_TP2)]


@requires_gfx950
@pytest.mark.parametrize("case_id,graphs,tp", _DFLY_PARAMS)
def test_hyv4_dfly_greedy_matches_target(
    vllm_runner,
    hyv4_recipe_env,
    eager_target_outputs,
    case_id: str,
    graphs: bool,
    tp: int,
) -> None:
    """Greedy DFly must reproduce the eager target, run after run.

    Speculative decoding is lossless at temperature 0, so any divergence beyond
    a near tie points at the DFly draft/verify path, or with graphs at stale
    FULL_DECODE_ONLY replay metadata on padded decode batches.
    """
    case = _E2E_CASES[case_id]
    expected = _eager_target(vllm_runner, eager_target_outputs, case_id, tp)
    with vllm_runner(case.model, **case.runner_kwargs(graphs, True, tp)) as model:
        if graphs:
            assert all(n > 0 for n in model.collective_rpc(_num_captured_graphs))
        _reinit_head(model)
        first = _greedy(model, case)
        assert _num_drafts(model.llm) > 0, "DFly drafter did not run"
        second = _greedy(model, case)

    atol = case.logprob_atol
    _assert_greedy_close(first, second, "dfly", "dfly-rerun", atol)
    _assert_greedy_close(expected, first, "target-eager", "dfly", atol)


@requires_gfx950
@pytest.mark.parametrize(
    "case_id,tp",
    [pytest.param(case_id, 1, id=case_id) for case_id in _E2E_CASES]
    + [pytest.param("hy4", 2, id="tp2-hy4", marks=_TP2)],
)
def test_hyv4_full_decode_graphs_match_eager(
    vllm_runner, hyv4_recipe_env, eager_target_outputs, case_id: str, tp: int
) -> None:
    """FULL_DECODE_ONLY graph replay must decode like eager for the target."""
    case = _E2E_CASES[case_id]
    expected = _eager_target(vllm_runner, eager_target_outputs, case_id, tp)
    with vllm_runner(
        case.model, **case.runner_kwargs(graphs=True, dfly=False, tp=tp)
    ) as model:
        assert model.collective_rpc(_num_captured_graphs) == [4] * tp
        _reinit_head(model)
        actual = _greedy(model, case)

    _assert_greedy_close(
        expected, actual, "target-eager", "target-graphs", case.logprob_atol
    )


def _small_hyv4_flash_vl_config(config):
    config = _small_hyv4_flash_dfly_config(config)
    if hasattr(config, "vision_config"):
        config.vision_config.num_hidden_layers = 2
    return config


def _check_encoder_graph_replay(worker, batches):
    """Replay each image batch through the encoder graph manager and compare
    it with the eager vision tower; return the graph hits per batch."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    runner = worker.model_runner
    manager = runner.model_state.encoder_runner.cudagraph_manager
    assert manager is not None and manager.is_captured()
    model = runner.model
    assert model.visual.attn_backend == AttentionBackendEnum.ROCM_AITER_FA
    vision = model.config.vision_config
    width = vision.num_channels * vision.patch_size**2
    generator = torch.Generator(device=runner.device).manual_seed(0)
    hits = []
    with torch.inference_mode():
        for grids in batches:
            patches = sum(t * h * w for t, h, w in grids)
            kwargs = {
                "pixel_values": torch.randn(
                    patches, width, device=runner.device, generator=generator
                ).bfloat16(),
                "image_grid_thw": torch.tensor(grids, device=runner.device),
            }
            before = manager.get_cumulative_stats()["graph_hits"]
            actual = manager.execute(dict(kwargs))
            hits.append(manager.get_cumulative_stats()["graph_hits"] - before)
            expected = model.embed_multimodal(**kwargs)
            assert len(actual) == len(expected) == len(grids)
            for output, reference in zip(actual, expected):
                assert torch.isfinite(output).all() and reference.abs().max() > 0
                # Two BF16 ulps at the output scale: replay pads the packed run.
                bound = 2 * reference.abs().max().item() * 2**-7
                torch.testing.assert_close(output, reference, rtol=0, atol=bound)
    return hits


@requires_gfx950
def test_hyv4_flash_vl_encoder_graphs_match_eager(vllm_runner, hyv4_recipe_env) -> None:
    """ROCm ViT encoder CUDA graphs (ROCM_AITER_FA varlen attention) replay
    single and batched images like the eager tower; an image over the largest
    budget falls back to eager."""
    from functools import partial

    with vllm_runner(
        HY4_FLASH_FP8,
        load_format="dummy",
        dtype="bfloat16",
        hf_overrides=_small_hyv4_flash_vl_config,
        max_model_len=1024,
        enable_chunked_prefill=True,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        limit_mm_per_prompt={"image": 2, "video": 1},
        mm_encoder_attn_backend="ROCM_AITER_FA",
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=[256, 1024],
            encoder_cudagraph_max_vision_items_per_batch=2,
        ),
    ) as model:
        results = model.llm.collective_rpc(
            partial(
                _check_encoder_graph_replay,
                batches=[
                    [[1, 16, 16]],
                    [[1, 8, 16], [1, 16, 24]],
                    [[1, 48, 48]],
                ],
            )
        )
    assert results == [[1, 2, 0]]
