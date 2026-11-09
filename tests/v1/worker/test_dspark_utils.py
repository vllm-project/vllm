# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import ParallelConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _get_dspark_parallel_config,
)
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

TARGET_VOCAB = 16
HIDDEN_SIZE = 8


@dataclass
class _FakeHFConfig:
    model_type: str


@dataclass
class _FakeModelConfig:
    hf_config: _FakeHFConfig


class _FakeDraftModelConfig:
    def __init__(self, vocab_size: int, draft_vocab_size: int):
        self._vocab_size = vocab_size
        self.hf_config = SimpleNamespace(draft_vocab_size=draft_vocab_size)

    def get_vocab_size(self) -> int:
        return self._vocab_size


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


class _QuantizedModule(nn.Module):
    """Packed module shape used to exercise _should_share's no-.weight path."""

    def __init__(self):
        super().__init__()
        self.weight_packed = nn.Parameter(
            torch.ones(1, dtype=torch.int8), requires_grad=False
        )


class _DraftWrapper(nn.Module):
    def __init__(
        self,
        *,
        has_own_embed_tokens: bool = False,
        has_own_lm_head: bool = False,
        embed_tokens: nn.Module | None = None,
        lm_head: nn.Module | None = None,
    ):
        super().__init__()
        self.has_own_embed_tokens = has_own_embed_tokens
        self.has_own_lm_head = has_own_lm_head
        self.model = nn.Module()
        self.model.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.model.embed_tokens is not None
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.lm_head(hidden_states)


def _fake_pp(world_size: int = 1):
    return lambda: SimpleNamespace(world_size=world_size, is_first_rank=True)


def _embedding(fill: float | None = None, vocab_size: int = TARGET_VOCAB):
    embedding = nn.Embedding(vocab_size, HIDDEN_SIZE)
    if fill is not None:
        with torch.no_grad():
            embedding.weight.fill_(fill)
    return embedding


def _lm_head(fill: float | None = None, vocab_size: int = TARGET_VOCAB):
    head = nn.Linear(HIDDEN_SIZE, vocab_size, bias=False)
    if fill is not None:
        with torch.no_grad():
            head.weight.fill_(fill)
    return head


def _target_model(embed_tokens: nn.Module | None, lm_head: nn.Module | None):
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = embed_tokens
    target.lm_head = lm_head
    return target


def _run_dspark_vocab_case(monkeypatch, input_vocab_size: int, output_vocab_size: int):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp())
    target_embed = _embedding()
    target_head = _lm_head()
    target = _target_model(target_embed, target_head)
    draft = _DraftWrapper()
    error: Exception | None = None

    try:
        dspark_utils._share_dspark_target_modules(
            draft_model=draft,
            draft_model_config=_FakeDraftModelConfig(
                input_vocab_size, output_vocab_size
            ),
            target_model=target,
            target_language_model=target,
            target_vocab_size=TARGET_VOCAB,
            draft_lm_head_quantization=None,
        )
    except Exception as exc:
        error = exc

    first_forward_error: Exception | None = None
    if error is None:
        try:
            input_id = min(input_vocab_size, TARGET_VOCAB) - 1
            draft.embed_input_ids(torch.tensor([input_id]))
            draft.compute_logits(torch.ones(1, HIDDEN_SIZE))
        except Exception as exc:
            first_forward_error = exc

    return SimpleNamespace(
        draft=draft,
        target_embed=target_embed,
        target_head=target_head,
        loader_error=error,
        first_forward_error=first_forward_error,
    )


@pytest.mark.parametrize(
    ("model_type", "enable_eplb", "num_redundant_experts", "shares_eplb_config"),
    [
        ("deepseek_v4", True, 32, True),
        ("deepseek_v41", False, 0, False),
    ],
)
def test_dspark_parallel_config_eplb_by_model_type(
    model_type: str,
    enable_eplb: bool,
    num_redundant_experts: int,
    shares_eplb_config: bool,
):
    target_config = _FakeParallelConfig()
    draft_model_config = _FakeModelConfig(
        hf_config=_FakeHFConfig(model_type=model_type)
    )

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
        draft_model_config=draft_model_config,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert draft_config.enable_eplb is enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == num_redundant_experts
    assert not draft_config.enable_elastic_ep
    if shares_eplb_config:
        assert draft_config.eplb_config is target_config.eplb_config
    else:
        assert draft_config.eplb_config is not target_config.eplb_config


@pytest.mark.parametrize(
    (
        "input_vocab_size",
        "output_vocab_size",
        "expect_embed_shared",
        "expect_head_shared",
        "expect_loader_error",
    ),
    [
        pytest.param(16, 16, True, True, None, id="input16-output16"),
        pytest.param(8, 16, True, True, None, id="input8-output16"),
        pytest.param(24, 16, False, True, "shared embedding", id="input24-output16"),
        pytest.param(16, 8, True, False, "shared LM head", id="input16-output8"),
        pytest.param(16, 24, True, False, "shared LM head", id="input16-output24"),
        pytest.param(24, 24, False, False, "shared embedding", id="input24-output24"),
    ],
)
def test_dspark_vocab_gates_share_only_compatible_target_modules(
    monkeypatch,
    input_vocab_size,
    output_vocab_size,
    expect_embed_shared,
    expect_head_shared,
    expect_loader_error,
):
    result = _run_dspark_vocab_case(
        monkeypatch, input_vocab_size, output_vocab_size
    )

    assert (
        result.draft.model.embed_tokens is result.target_embed
    ) is expect_embed_shared
    assert (result.draft.lm_head is result.target_head) is expect_head_shared
    if expect_loader_error is None:
        assert result.loader_error is None
        assert result.first_forward_error is None
    else:
        assert isinstance(result.loader_error, ValueError)
        assert expect_loader_error in str(result.loader_error)
        assert f"target vocab_size={TARGET_VOCAB}" in str(result.loader_error)


@pytest.mark.parametrize(
    ("has_own_embed_tokens", "has_own_lm_head"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_dspark_ownership_combinations_preserve_share_when_weights_match(
    monkeypatch, has_own_embed_tokens, has_own_lm_head
):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp())
    target_embed = _embedding(fill=1.0)
    target_head = _lm_head(fill=1.0)
    draft = _DraftWrapper(
        has_own_embed_tokens=has_own_embed_tokens,
        has_own_lm_head=has_own_lm_head,
        embed_tokens=_embedding(fill=1.0),
        lm_head=_lm_head(fill=1.0),
    )

    eagle_utils.maybe_share_target_embed(
        draft, draft.model, _target_model(target_embed, target_head).model
    )
    eagle_utils.share_target_lm_head(
        draft, target_head, quantization=None, can_share=True
    )

    assert draft.model.embed_tokens is target_embed
    assert draft.lm_head is target_head


@pytest.mark.parametrize(
    ("draft_embed", "expect_shared"),
    [
        pytest.param(None, False, id="missing"),
        pytest.param(_embedding(fill=1.0), True, id="matching"),
        pytest.param(_embedding(fill=2.0), False, id="different"),
        pytest.param(_QuantizedModule(), False, id="quantized"),
    ],
)
def test_owned_embedding_is_only_replaced_when_it_matches_target(
    monkeypatch, draft_embed, expect_shared
):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp())
    target_embed = _embedding(fill=1.0)
    draft = _DraftWrapper(has_own_embed_tokens=True, embed_tokens=draft_embed)

    eagle_utils.maybe_share_target_embed(
        draft, draft.model, _target_model(target_embed, None).model
    )

    assert (draft.model.embed_tokens is target_embed) is expect_shared
    if not expect_shared:
        assert draft.model.embed_tokens is draft_embed


@pytest.mark.parametrize(
    ("draft_head", "expect_shared"),
    [
        pytest.param(None, False, id="missing"),
        pytest.param(_lm_head(fill=1.0), True, id="matching"),
        pytest.param(_lm_head(fill=2.0), False, id="different"),
        pytest.param(_QuantizedModule(), False, id="quantized"),
    ],
)
def test_owned_lm_head_is_only_replaced_when_it_matches_target(
    draft_head, expect_shared
):
    target_head = _lm_head(fill=1.0)
    draft = _DraftWrapper(has_own_lm_head=True, lm_head=draft_head)

    result = eagle_utils.share_target_lm_head(
        draft, target_head, quantization=None, can_share=True
    )

    assert (draft.lm_head is target_head) is expect_shared
    assert (result is target_head) is expect_shared
    if not expect_shared:
        assert draft.lm_head is draft_head


def test_missing_target_embedding_raises_when_dspark_needs_share(monkeypatch):
    monkeypatch.setattr(eagle_utils, "get_pp_group", _fake_pp())
    target_inner = nn.Module()
    target_inner.embed_tokens = None
    target_inner.embedding = None
    draft = _DraftWrapper(has_own_embed_tokens=False)

    with pytest.raises(RuntimeError, match="needs the target input embedding"):
        eagle_utils.maybe_share_target_embed(draft, draft.model, target_inner)


def test_missing_target_lm_head_leaves_unowned_dspark_head_unset():
    draft = _DraftWrapper(has_own_lm_head=False)

    result = eagle_utils.share_target_lm_head(
        draft, target_lm_head=None, quantization=None, can_share=True
    )

    assert result is None
    assert draft.lm_head is None


def test_missing_target_lm_head_with_quantization_keeps_existing_error():
    draft = _DraftWrapper(has_own_lm_head=False)

    with pytest.raises(ValueError, match="draft_lm_head_quantization"):
        eagle_utils.share_target_lm_head(
            draft, target_lm_head=None, quantization="fp8", can_share=True
        )


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("use_mla", [False, True])
def test_draft_context_parallelism_without_changing_target(
    monkeypatch, pcp_size, dcp_size, use_mla
):
    target_parallel = ParallelConfig(
        tensor_parallel_size=4,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(use_mla=use_mla)
        ),
    )

    class CapturedConfig(Exception):
        pass

    def capture_init(self, config, device):
        raise CapturedConfig(config)

    monkeypatch.setattr(DraftModelSpeculator, "__init__", capture_init)
    with pytest.raises(CapturedConfig) as captured:
        DFlashSpeculator(target_config, device=None)
    draft_parallel = captured.value.args[0].parallel_config
    assert draft_parallel.tensor_parallel_size == 4
    assert draft_parallel.prefill_context_parallel_size == 1
    assert draft_parallel.decode_context_parallel_size == (dcp_size if use_mla else 1)
    assert draft_parallel.cp_kv_cache_interleave_size == 16
    assert target_parallel.prefill_context_parallel_size == pcp_size
    assert target_parallel.decode_context_parallel_size == dcp_size
    assert target_parallel.cp_kv_cache_interleave_size == 16


@pytest.mark.parametrize("configured_dcp", [1, 4])
def test_attention_uses_draft_dcp_setting_inside_target_process_group(
    monkeypatch, configured_dcp
):
    import vllm.config as config_module
    from vllm.distributed import parallel_state
    from vllm.v1.attention.backend import AttentionImplBase, AttentionMetadata

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=configured_dcp)
    )
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )

    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=4, rank_in_group=2)
    )
    impl: AttentionImplBase[AttentionMetadata] = AttentionImplBase()
    assert impl.dcp_world_size == configured_dcp
    assert impl.dcp_rank == (0 if configured_dcp == 1 else 2)


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_runner_marks_only_replicated_draft_caches(monkeypatch, draft_dcp_size):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        model_runner, "get_kv_cache_spec", lambda _: {"target": spec, "draft": spec}
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    specs = runner.get_kv_cache_spec()

    assert specs["target"].dcp_sharded
    assert specs["draft"].dcp_sharded == (draft_dcp_size == 4)
