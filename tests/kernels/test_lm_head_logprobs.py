# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def _require_supported_cutedsl_device() -> None:
    pytest.importorskip("cutlass")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Requires SM80+")
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    max_smem = getattr(
        props, "shared_memory_per_block_optin", props.shared_memory_per_block
    )
    if max_smem < 144 * 1024:
        pytest.skip("Requires at least 144 KiB of shared memory per block")


@pytest.mark.usefixtures("_require_supported_cutedsl_device")
@pytest.mark.parametrize("num_topk", [0, 1, 5, 20, 32])
def test_lm_head_logprobs_matches_full_projection(num_topk: int) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        lm_head_logprobs,
    )

    generator = torch.Generator(device="cuda").manual_seed(34)
    hidden_states = (
        torch.randn((17, 72), device="cuda", dtype=torch.bfloat16, generator=generator)
        * 0.01
    )
    lm_head_weight = (
        torch.randn(
            (1051, 72), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.01
    )
    local_target_ids = torch.tensor(
        [0, 511, 512, 1024, 1049] + [-1] * 12,
        device="cuda",
        dtype=torch.int32,
    )
    target_logits = torch.linspace(-0.25, 0.25, 17, device="cuda", dtype=torch.float32)
    valid_vocab_size = 1050
    global_vocab_start = 2000

    output = lm_head_logprobs(
        hidden_states,
        lm_head_weight,
        local_target_ids,
        target_logits,
        num_topk,
        valid_vocab_size=valid_vocab_size,
        global_vocab_start=global_vocab_start,
    )

    logits = hidden_states.float() @ lm_head_weight[:valid_vocab_size].float().T
    local_target_mask = local_target_ids >= 0
    rows = torch.arange(hidden_states.shape[0], device="cuda")[local_target_mask]
    logits[rows, local_target_ids[local_target_mask].long()] = target_logits[
        local_target_mask
    ]
    expected_lse = torch.logsumexp(logits, dim=-1)
    expected_rank = (logits >= target_logits[:, None]).sum(dim=-1).to(torch.int32)

    assert torch.equal(output.rank_count, expected_rank)
    assert torch.allclose(output.lse, expected_lse, rtol=2e-3, atol=2e-3)
    if num_topk == 0:
        assert output.topk_values.shape == (17, 0)
        assert output.topk_ids.shape == (17, 0)
        return

    expected_values, expected_ids = torch.topk(logits, num_topk, dim=-1)
    expected_ids = expected_ids.to(torch.int32) + global_vocab_start
    assert torch.equal(output.topk_ids, expected_ids)
    assert torch.allclose(output.topk_values, expected_values, rtol=2e-3, atol=2e-3)


@pytest.mark.usefixtures("_require_supported_cutedsl_device")
def test_lm_head_logprobs_preserves_empty_topk_width() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        lm_head_logprobs,
    )

    output = lm_head_logprobs(
        torch.empty((0, 64), device="cuda", dtype=torch.bfloat16),
        torch.empty((128, 64), device="cuda", dtype=torch.bfloat16),
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.float32),
        5,
    )

    assert output.topk_values.shape == (0, 5)
    assert output.topk_ids.shape == (0, 5)
    assert output.lse.shape == (0,)
    assert output.rank_count.shape == (0,)


@pytest.mark.usefixtures("_require_supported_cutedsl_device")
def test_lm_head_logprobs_handles_ties_padding_and_rank_boundaries() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        lm_head_logprobs,
    )

    hidden_states = torch.ones((2, 64), device="cuda", dtype=torch.bfloat16)
    lm_head_weight = torch.zeros((260, 64), device="cuda", dtype=torch.bfloat16)
    # Padding logits would dominate if valid_vocab_size were not respected.
    lm_head_weight[257:] = 8
    local_target_ids = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    target_logits = torch.tensor([0.0, 1.0], device="cuda", dtype=torch.float32)

    output = lm_head_logprobs(
        hidden_states,
        lm_head_weight,
        local_target_ids,
        target_logits,
        5,
        valid_vocab_size=257,
        global_vocab_start=100,
    )

    expected_ids = torch.tensor(
        [[100, 101, 102, 103, 104], [356, 100, 101, 102, 103]],
        device="cuda",
        dtype=torch.int32,
    )
    expected_values = torch.tensor(
        [[0.0] * 5, [1.0, 0.0, 0.0, 0.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    expected_lse = torch.tensor(
        [math.log(257.0), math.log(math.e + 256.0)],
        device="cuda",
        dtype=torch.float32,
    )

    assert torch.equal(output.topk_ids, expected_ids)
    assert torch.equal(output.topk_values, expected_values)
    assert torch.equal(
        output.rank_count,
        torch.tensor([257, 1], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(output.lse, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("tp_size", "num_topk"),
    [(1, 0), (2, 5), (4, 20)],
)
def test_merge_tp_prompt_logprobs_matches_full_logits(
    tp_size: int, num_topk: int
) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        merge_tp_prompt_logprobs,
    )

    generator = torch.Generator(device="cuda").manual_seed(61)
    num_rows, local_vocab_size = 7, 37
    logits = torch.randn(
        (tp_size, num_rows, local_vocab_size),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    global_logits = logits.permute(1, 0, 2).reshape(num_rows, -1)
    target_token_ids = torch.linspace(
        0,
        global_logits.shape[1] - 1,
        num_rows,
        device="cuda",
        dtype=torch.int64,
    )
    target_logits = global_logits.gather(1, target_token_ids[:, None]).squeeze(1)

    local_values = []
    local_ids = []
    local_lse = []
    local_rank_count = []
    for tp_rank in range(tp_size):
        values, ids = torch.topk(logits[tp_rank], num_topk, dim=1)
        local_values.append(values)
        local_ids.append((ids + tp_rank * local_vocab_size).to(torch.int32))
        local_lse.append(torch.logsumexp(logits[tp_rank], dim=1))
        local_rank_count.append(
            (logits[tp_rank] >= target_logits[:, None]).sum(dim=1, dtype=torch.int32)
        )

    output_ids, output_logprobs, output_ranks = merge_tp_prompt_logprobs(
        torch.stack(local_values),
        torch.stack(local_ids),
        torch.stack(local_lse),
        torch.stack(local_rank_count),
        target_token_ids,
        target_logits,
        num_topk,
    )

    global_lse = torch.logsumexp(global_logits, dim=1)
    expected_ids = target_token_ids[:, None].to(torch.int32)
    expected_logprobs = target_logits[:, None] - global_lse[:, None]
    if num_topk > 0:
        topk_values, topk_ids = torch.topk(global_logits, num_topk, dim=1)
        expected_ids = torch.cat((expected_ids, topk_ids.to(torch.int32)), dim=1)
        expected_logprobs = torch.cat(
            (expected_logprobs, topk_values - global_lse[:, None]), dim=1
        )
    expected_ranks = (global_logits >= target_logits[:, None]).sum(
        dim=1, dtype=torch.int32
    )

    assert torch.equal(output_ids, expected_ids)
    assert torch.equal(output_ranks, expected_ranks)
    assert torch.allclose(output_logprobs, expected_logprobs, atol=1e-5, rtol=1e-5)


def test_merge_tp_prompt_logprobs_orders_cross_rank_ties() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        merge_tp_prompt_logprobs,
    )

    local_logits = torch.tensor(
        [[[2.0, 2.0, 1.0]], [[2.0, 1.5, 1.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    local_ids = torch.tensor(
        [[[5, 7, 9]], [[2, 8, 10]]],
        device="cuda",
        dtype=torch.int32,
    )
    local_lse = torch.logsumexp(local_logits, dim=-1)
    output_ids, _, output_ranks = merge_tp_prompt_logprobs(
        local_logits,
        local_ids,
        local_lse,
        torch.tensor([[3], [3]], device="cuda", dtype=torch.int32),
        torch.tensor([10], device="cuda", dtype=torch.int64),
        torch.tensor([1.0], device="cuda", dtype=torch.float32),
        3,
    )

    assert torch.equal(
        output_ids,
        torch.tensor([[10, 2, 5, 7]], device="cuda", dtype=torch.int32),
    )
    assert output_ranks.item() == 6


def _prompt_target_logits_reference(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    local_target_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output = torch.zeros(
        hidden_states.shape[0],
        dtype=torch.float32,
        device=hidden_states.device,
    )
    is_local = (local_target_ids >= 0) & (local_target_ids < lm_head_weight.shape[0])
    if not torch.any(is_local):
        return output

    target_ids = local_target_ids[is_local].to(torch.int64)
    output[is_local] = (
        hidden_states[is_local].float()
        * lm_head_weight.index_select(0, target_ids).float()
    ).sum(dim=-1)
    if bias is not None:
        output[is_local] += bias.index_select(0, target_ids).float()
    return output


def test_prompt_target_logits_non_local_targets_contribute_zero() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        prompt_target_logits,
    )

    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden_states = torch.randn(
        (7, 497),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    lm_head_weight = torch.randn(
        (19, 497),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    bias = torch.randn(
        19,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    local_target_ids = torch.tensor(
        [0, 18, -1, 19, 7, 3, 12],
        device="cuda",
        dtype=torch.int64,
    )

    actual = prompt_target_logits(
        hidden_states,
        lm_head_weight,
        local_target_ids,
        bias,
    )
    expected = _prompt_target_logits_reference(
        hidden_states,
        lm_head_weight,
        local_target_ids,
        bias,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
    assert actual[2].item() == 0.0
    assert actual[3].item() == 0.0


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        pytest.param(torch.float16, 2e-3, id="fp16"),
        pytest.param(torch.bfloat16, 2e-2, id="bf16"),
    ],
)
def test_prompt_target_logits_fp32_accumulation(
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        prompt_target_logits,
    )

    generator = torch.Generator(device="cuda").manual_seed(1)
    hidden_states = torch.randn(
        (11, 4097),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    lm_head_weight = torch.randn(
        (23, 4097),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    local_target_ids = torch.arange(11, device="cuda", dtype=torch.int32)

    actual = prompt_target_logits(
        hidden_states,
        lm_head_weight,
        local_target_ids,
    )
    expected = _prompt_target_logits_reference(
        hidden_states,
        lm_head_weight,
        local_target_ids,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def test_prompt_target_logits_shard_sum_recovers_global_logits() -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        prompt_target_logits,
    )

    generator = torch.Generator(device="cuda").manual_seed(2)
    hidden_states = torch.randn(
        (6, 513),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    full_weight = torch.randn(
        (20, 513),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    full_bias = torch.randn(
        20,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    global_target_ids = torch.tensor(
        [0, 9, 10, 11, 18, 19],
        device="cuda",
        dtype=torch.int64,
    )
    rank_0_ids = torch.where(global_target_ids < 10, global_target_ids, -1)
    rank_1_ids = torch.where(
        global_target_ids >= 10,
        global_target_ids - 10,
        -1,
    )

    rank_0_logits = prompt_target_logits(
        hidden_states,
        full_weight[:10],
        rank_0_ids,
        full_bias[:10],
    )
    rank_1_logits = prompt_target_logits(
        hidden_states,
        full_weight[10:],
        rank_1_ids,
        full_bias[10:],
    )

    expected = (
        hidden_states.float() * full_weight.index_select(0, global_target_ids).float()
    ).sum(dim=-1)
    expected += full_bias.index_select(0, global_target_ids).float()
    torch.testing.assert_close(
        rank_0_logits + rank_1_logits,
        expected,
        rtol=1e-5,
        atol=1e-4,
    )


def test_prompt_logprobs_warmup_compiles_cute_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.utils import ensure_current_vllm_config
    from vllm.model_executor.kernels.linear.cute_dsl import lm_head_logprobs
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    monkeypatch.setattr(
        lm_head_logprobs,
        "validate_lm_head_logprobs_environment",
        lambda _weight: None,
    )
    compiled_k = []

    with ensure_current_vllm_config():
        processor = LogitsProcessor(128)

    def record_k(*_args, num_logprobs: int | None = None):
        # get_prompt_logprobs receives K as its fourth positional argument.
        compiled_k.append(_args[-1] if num_logprobs is None else num_logprobs)

    monkeypatch.setattr(processor, "get_prompt_logprobs", record_k)
    processor.warmup_prompt_logprobs(
        SimpleNamespace(
            weight=torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
        )
    )

    assert compiled_k == [0, 32]


def test_prompt_logprobs_warmup_reports_environment_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.utils import ensure_current_vllm_config
    from vllm.model_executor.kernels.linear.cute_dsl import lm_head_logprobs
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    def reject_environment(_weight: torch.Tensor) -> None:
        raise RuntimeError(
            "the current GPU exposes only 128 KiB shared memory; "
            "the compact prompt-logprobs path requires 144 KiB"
        )

    monkeypatch.setattr(
        lm_head_logprobs,
        "validate_lm_head_logprobs_environment",
        reject_environment,
    )
    with ensure_current_vllm_config():
        processor = LogitsProcessor(128)

    with pytest.raises(
        RuntimeError,
        match=(
            "VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS is enabled, but the current "
            "GPU exposes only 128 KiB shared memory"
        ),
    ):
        processor.warmup_prompt_logprobs(
            SimpleNamespace(
                weight=torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
            )
        )


def test_prompt_logprobs_warmup_reports_missing_cutedsl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    from tests.utils import ensure_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    module_name = "vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs"
    monkeypatch.setitem(sys.modules, module_name, None)
    with ensure_current_vllm_config():
        processor = LogitsProcessor(128)

    with pytest.raises(
        RuntimeError, match="CuTe DSL dependencies could not be imported"
    ):
        processor.warmup_prompt_logprobs(
            SimpleNamespace(
                weight=torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
            )
        )


@pytest.mark.parametrize(
    ("processor_updates", "lm_head_updates", "hidden_dtype", "error", "match"),
    [
        ({"logits_as_input": True}, {}, torch.bfloat16, ValueError, "projection"),
        ({"scale": 2.0}, {}, torch.bfloat16, ValueError, "unmodified"),
        ({"soft_cap": 10.0}, {}, torch.bfloat16, ValueError, "unmodified"),
        ({"head_dtype": torch.float32}, {}, torch.bfloat16, ValueError, "FP32"),
        ({}, {"quant_method": object()}, torch.bfloat16, TypeError, "unquantized"),
        ({}, {}, torch.float16, TypeError, "BF16 hidden states"),
        (
            {},
            {"weight": torch.empty((128, 64), dtype=torch.float16)},
            torch.bfloat16,
            TypeError,
            "BF16 hidden states",
        ),
        (
            {},
            {"bias": torch.empty(128)},
            torch.bfloat16,
            ValueError,
            "LM-head bias",
        ),
        (
            {},
            {"num_added_embeddings": 1},
            torch.bfloat16,
            ValueError,
            "added vocabulary",
        ),
        (
            {},
            {"org_vocab_size": 129},
            torch.bfloat16,
            ValueError,
            "vocabulary sizes must match",
        ),
    ],
)
def test_validate_prompt_logprobs_rejects_unsupported_config(
    processor_updates: dict[str, object],
    lm_head_updates: dict[str, object],
    hidden_dtype: torch.dtype,
    error: type[Exception],
    match: str,
) -> None:
    from types import SimpleNamespace

    from tests.utils import ensure_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    with ensure_current_vllm_config():
        processor = LogitsProcessor(128)
    for name, value in processor_updates.items():
        setattr(processor, name, value)

    lm_head_attributes = {
        "weight": torch.empty((128, 64), dtype=torch.bfloat16),
        "bias": None,
        "quant_method": UnquantizedEmbeddingMethod(),
        "num_added_embeddings": 0,
        "org_vocab_size": 128,
    }
    lm_head_attributes.update(lm_head_updates)
    lm_head = SimpleNamespace(**lm_head_attributes)

    with pytest.raises(error, match=match):
        processor.validate_prompt_logprobs(lm_head, hidden_dtype)


@pytest.mark.parametrize(
    ("weight_layout", "match"),
    [
        ("noncontiguous", "contiguous"),
        ("hidden_size", "hidden size must be divisible by 8"),
        ("unaligned", "16-byte aligned"),
    ],
)
def test_validate_lm_head_logprobs_environment_rejects_invalid_weight(
    weight_layout: str,
    match: str,
) -> None:
    pytest.importorskip("cutlass")
    from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
        validate_lm_head_logprobs_environment,
    )

    if weight_layout == "noncontiguous":
        weight = torch.empty((64, 128), device="cuda", dtype=torch.bfloat16).T
    elif weight_layout == "hidden_size":
        weight = torch.empty((128, 65), device="cuda", dtype=torch.bfloat16)
    else:
        storage = torch.empty(128 * 64 + 1, device="cuda", dtype=torch.bfloat16)
        weight = storage[1:].view(128, 64)

    with pytest.raises(ValueError, match=match):
        validate_lm_head_logprobs_environment(weight)


def test_validate_lm_head_logprobs_environment_rejects_pre_sm80(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    pytest.importorskip("cutlass")
    from vllm.model_executor.kernels.linear.cute_dsl import lm_head_logprobs

    capability = SimpleNamespace(major=7, to_int=lambda: 70)
    monkeypatch.setattr(
        lm_head_logprobs.current_platform,
        "get_device_capability",
        lambda _device_index: capability,
    )
    weight = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="requires SM80 or newer"):
        lm_head_logprobs.validate_lm_head_logprobs_environment(weight)


def test_validate_lm_head_logprobs_environment_rejects_insufficient_smem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    pytest.importorskip("cutlass")
    from vllm.model_executor.kernels.linear.cute_dsl import lm_head_logprobs

    capability = SimpleNamespace(major=8, to_int=lambda: 80)
    monkeypatch.setattr(
        lm_head_logprobs.current_platform,
        "get_device_capability",
        lambda _device_index: capability,
    )
    monkeypatch.setattr(
        lm_head_logprobs,
        "cuda_get_device_properties",
        lambda *_args, **_kwargs: (128 * 1024,),
    )
    weight = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="requires 144 KiB"):
        lm_head_logprobs.validate_lm_head_logprobs_environment(weight)


@pytest.mark.parametrize("num_logprobs", [0, 5, 20])
@pytest.mark.usefixtures("_require_supported_cutedsl_device")
def test_logits_processor_prompt_logprobs_tp1(num_logprobs: int) -> None:
    """Validate the compact FP32-accumulator prompt-logprobs contract."""
    from types import SimpleNamespace

    from tests.utils import ensure_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    generator = torch.Generator(device="cuda").manual_seed(73)
    num_rows, hidden_size, vocab_size = 17, 72, 1050
    hidden_states = (
        torch.randn(
            (num_rows, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    lm_head_weight = (
        torch.randn(
            (vocab_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.01
    )
    target_token_ids = torch.linspace(
        0,
        vocab_size - 1,
        num_rows,
        device="cuda",
        dtype=torch.int64,
    )
    shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=vocab_size,
        num_org_elements=vocab_size,
    )
    lm_head = SimpleNamespace(
        weight=lm_head_weight,
        bias=None,
        quant_method=UnquantizedEmbeddingMethod(),
        num_added_embeddings=0,
        org_vocab_size=vocab_size,
        tp_size=1,
        shard_indices=shard_indices,
    )

    with ensure_current_vllm_config():
        output_ids, output_logprobs, output_ranks = LogitsProcessor(
            vocab_size
        ).get_prompt_logprobs(
            lm_head,
            hidden_states,
            target_token_ids,
            num_logprobs,
        )

    # Compact prompt logprobs intentionally keep FP32 accumulation through
    # top-K, LSE, and rank computation; this is its numerical reference.
    logits = hidden_states.float() @ lm_head_weight.float().T
    target_logits = logits.gather(1, target_token_ids[:, None]).squeeze(1)
    lse = torch.logsumexp(logits, dim=1)
    expected_ids = target_token_ids[:, None].to(torch.int32)
    expected_logprobs = target_logits[:, None] - lse[:, None]
    if num_logprobs > 0:
        topk_values, topk_ids = torch.topk(logits, num_logprobs, dim=1)
        expected_ids = torch.cat((expected_ids, topk_ids.to(torch.int32)), dim=1)
        expected_logprobs = torch.cat(
            (expected_logprobs, topk_values - lse[:, None]), dim=1
        )
    expected_ranks = (logits >= target_logits[:, None]).sum(dim=1, dtype=torch.int32)

    assert torch.equal(output_ids, expected_ids)
    assert torch.equal(output_ranks, expected_ranks)
    torch.testing.assert_close(
        output_logprobs,
        expected_logprobs,
        rtol=2e-3,
        atol=2e-3,
    )


def _run_logits_processor_prompt_logprobs_tp2(
    local_rank: int,
    world_size: int,
    master_port: int,
) -> None:
    from types import SimpleNamespace

    from tests.utils import ensure_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.utils.system_utils import update_environment_variables

    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    init_distributed_environment()
    try:
        with ensure_current_vllm_config():
            initialize_model_parallel(tensor_model_parallel_size=world_size)

            generator = torch.Generator(device=device).manual_seed(79)
            num_rows, hidden_size, vocab_size = 7, 64, 2048
            local_vocab_size = vocab_size // world_size
            hidden_states = (
                torch.randn(
                    (num_rows, hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                * 0.01
            )
            full_weight = (
                torch.randn(
                    (vocab_size, hidden_size),
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                * 0.01
            )
            vocab_start = local_rank * local_vocab_size
            vocab_end = vocab_start + local_vocab_size
            local_weight = full_weight[vocab_start:vocab_end]
            target_token_ids = torch.tensor(
                [0, 511, 1023, 1024, 1535, 1536, 2047],
                device=device,
                dtype=torch.int64,
            )
            shard_indices = SimpleNamespace(
                org_vocab_start_index=vocab_start,
                org_vocab_end_index=vocab_end,
                num_org_elements=local_vocab_size,
            )
            lm_head = SimpleNamespace(
                weight=local_weight,
                bias=None,
                quant_method=UnquantizedEmbeddingMethod(),
                num_added_embeddings=0,
                org_vocab_size=vocab_size,
                tp_size=world_size,
                shard_indices=shard_indices,
            )
            processor = LogitsProcessor(vocab_size)
            logits = hidden_states.float() @ full_weight.float().T
            target_logits = logits.gather(1, target_token_ids[:, None]).squeeze(1)
            lse = torch.logsumexp(logits, dim=1)

            for num_logprobs in (0, 20):
                output_ids, output_logprobs, output_ranks = (
                    processor.get_prompt_logprobs(
                        lm_head,
                        hidden_states,
                        target_token_ids,
                        num_logprobs,
                    )
                )
                expected_ids = target_token_ids[:, None].to(torch.int32)
                expected_logprobs = target_logits[:, None] - lse[:, None]
                if num_logprobs > 0:
                    topk_values, topk_ids = torch.topk(logits, num_logprobs, dim=1)
                    expected_ids = torch.cat(
                        (expected_ids, topk_ids.to(torch.int32)), dim=1
                    )
                    expected_logprobs = torch.cat(
                        (expected_logprobs, topk_values - lse[:, None]), dim=1
                    )
                expected_ranks = (logits >= target_logits[:, None]).sum(
                    dim=1, dtype=torch.int32
                )

                assert torch.equal(output_ids, expected_ids)
                assert torch.equal(output_ranks, expected_ranks)
                torch.testing.assert_close(
                    output_logprobs,
                    expected_logprobs,
                    rtol=2e-3,
                    atol=2e-3,
                )
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.usefixtures("_require_supported_cutedsl_device")
def test_logits_processor_prompt_logprobs_tp2() -> None:
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip("Requires two GPUs")
    mp.spawn(
        _run_logits_processor_prompt_logprobs_tp2,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )
