# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.models.qwen4_exp.nvidia.fp8_proposal_head import Fp8ProposalHead
from vllm.models.qwen4_exp.nvidia.mtp import (
    Qwen4ExpMTP,
    _validate_fp8_proposal_head_config,
)
from vllm.platforms import current_platform


class _Head(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedEmbeddingMethod()
        self.parallel_group = None
        self.tp_size = 1
        self.shard_indices = SimpleNamespace(num_org_vocab_padding=0)
        self.num_embeddings_per_partition = weight.shape[0]
        self.embedding_dim = weight.shape[1]


def _empty_qwen_mtp(target_head: nn.Module) -> Qwen4ExpMTP:
    model = object.__new__(Qwen4ExpMTP)
    nn.Module.__init__(model)
    model.lm_head = target_head
    model._fp8_proposal_head = None
    return model


def _supported_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_v2_model_runner=True,
        lora_config=None,
        weight_transfer_config=None,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            head_dtype=torch.bfloat16,
            enforce_eager=True,
            enable_sleep_mode=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )


def test_fp8_proposal_head_config_accepts_qualified_scope() -> None:
    with patch("vllm.models.qwen4_exp.nvidia.mtp.envs.VLLM_BATCH_INVARIANT", False):
        _validate_fp8_proposal_head_config(_supported_config())


@pytest.mark.parametrize(
    ("owner", "field", "value", "message"),
    [
        ("config", "use_v2_model_runner", False, "Model Runner V1"),
        ("model", "dtype", torch.float16, "model dtype"),
        ("model", "head_dtype", torch.float32, "head dtype"),
        ("model", "enforce_eager", False, "CUDA graphs"),
        ("config", "lora_config", object(), "LoRA"),
        ("model", "enable_sleep_mode", True, "sleep mode"),
        ("config", "weight_transfer_config", object(), "weight transfer"),
        ("parallel", "tensor_parallel_size", 4, "tensor parallel size 4"),
        ("parallel", "pipeline_parallel_size", 2, "pipeline parallelism"),
        ("parallel", "prefill_context_parallel_size", 2, "prefill context"),
        ("parallel", "decode_context_parallel_size", 2, "decode context"),
    ],
)
def test_fp8_proposal_head_config_rejects_unqualified_scope(
    owner: str, field: str, value: object, message: str
) -> None:
    config = _supported_config()
    target = {
        "config": config,
        "model": config.model_config,
        "parallel": config.parallel_config,
    }[owner]
    setattr(target, field, value)

    with (
        patch(
            "vllm.models.qwen4_exp.nvidia.mtp.envs.VLLM_BATCH_INVARIANT",
            False,
        ),
        pytest.raises(ValueError, match=message),
    ):
        _validate_fp8_proposal_head_config(config)


def test_fp8_proposal_head_config_rejects_batch_invariance() -> None:
    with (
        patch(
            "vllm.models.qwen4_exp.nvidia.mtp.envs.VLLM_BATCH_INVARIANT",
            True,
        ),
        pytest.raises(ValueError, match="batch-invariant"),
    ):
        _validate_fp8_proposal_head_config(_supported_config())


def test_compute_logits_selects_private_head() -> None:
    target_head = _Head(torch.zeros(16, 16))
    proposal_head = Mock()
    model = _empty_qwen_mtp(target_head)
    model.logits_processor = Mock(side_effect=lambda head, hidden: (head, hidden))
    hidden = torch.zeros(1, 16)

    selected_head, selected_hidden = model.compute_logits(hidden, spec_step_idx=3)
    assert selected_head is target_head
    assert selected_hidden is hidden

    model._fp8_proposal_head = proposal_head
    selected_head, selected_hidden = model.compute_logits(hidden, spec_step_idx=0)
    assert selected_head is proposal_head
    assert selected_hidden is hidden
    assert model.lm_head is target_head


def test_initialization_is_disabled_idempotent_and_preserves_target() -> None:
    target_head = _Head(torch.randn(16, 16, dtype=torch.bfloat16))
    model = _empty_qwen_mtp(target_head)
    proposal_head = Mock(storage_bytes=272)
    target_identity = (
        id(model.lm_head),
        id(model.lm_head.weight),
        model.lm_head.weight.data_ptr(),
        id(model.lm_head.quant_method),
    )

    with (
        patch(
            "vllm.models.qwen4_exp.nvidia.mtp.envs.VLLM_QWEN4_EXP_FP8_DRAFT_HEAD",
            False,
        ),
        patch(
            "vllm.models.qwen4_exp.nvidia.mtp.Fp8ProposalHead",
            return_value=proposal_head,
        ) as create,
    ):
        model.maybe_init_fp8_proposal_head()
    create.assert_not_called()

    with (
        patch(
            "vllm.models.qwen4_exp.nvidia.mtp.envs.VLLM_QWEN4_EXP_FP8_DRAFT_HEAD",
            True,
        ),
        patch(
            "vllm.models.qwen4_exp.nvidia.mtp.Fp8ProposalHead",
            return_value=proposal_head,
        ) as create,
    ):
        model.maybe_init_fp8_proposal_head()
        model.maybe_init_fp8_proposal_head()

    create.assert_called_once_with(target_head)
    assert model._fp8_proposal_head is proposal_head
    assert target_identity == (
        id(model.lm_head),
        id(model.lm_head.weight),
        model.lm_head.weight.data_ptr(),
        id(model.lm_head.quant_method),
    )


def test_direct_reload_rejected_before_consuming_weights() -> None:
    model = _empty_qwen_mtp(_Head(torch.zeros(16, 16)))
    model._fp8_proposal_head = Mock()
    consumed = False

    def weights():
        nonlocal consumed
        consumed = True
        yield "weight", torch.zeros(1)

    with pytest.raises(RuntimeError, match="Cannot reload"):
        model.load_weights(weights())
    assert not consumed


def test_target_reload_rejected_after_initialization() -> None:
    model = _empty_qwen_mtp(_Head(torch.zeros(16, 16)))
    model._fp8_proposal_head = Mock()

    with pytest.raises(RuntimeError, match="Cannot reload target weights"):
        model.before_target_model_reload()


def test_helper_rejects_non_cuda_source() -> None:
    source = _Head(torch.zeros(16, 16, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        Fp8ProposalHead(source)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.supports_fp8(),
    reason="requires CUDA FP8 support",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
def test_fp8_proposal_head_cuda_math_and_ownership(
    num_tokens: int,
    default_vllm_config,
) -> None:
    torch.manual_seed(1234)
    weight = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
    weight.mul_(0.02)
    weight[0].zero_()
    weight[1:5].mul_(torch.tensor([0.01, 0.1, 10.0, 100.0], device="cuda")[:, None])
    source = _Head(weight)
    source_weight_id = id(source.weight)
    source_weight_ptr = source.weight.data_ptr()
    source_before = source.weight.clone()
    hidden = torch.randn(num_tokens, 256, dtype=torch.bfloat16, device="cuda")

    proposal = Fp8ProposalHead(source)
    actual = proposal.quant_method.apply(proposal, hidden)
    hidden_fp8, hidden_scale = ops.scaled_fp8_quant(
        hidden, use_per_token_if_dynamic=True
    )
    reference = (hidden_fp8.float() * hidden_scale.float()) @ (
        proposal.weight_fp8.float() * proposal.weight_scale.float()
    ).t()

    assert actual.shape == (num_tokens, 512)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), reference, rtol=0.03, atol=0.25)
    assert proposal.weight_fp8.dtype == torch.float8_e4m3fn
    assert proposal.weight_scale.dtype == torch.float32
    assert proposal.weight_fp8.untyped_storage().data_ptr() != source_weight_ptr
    assert list(proposal.named_parameters()) == []
    assert proposal.state_dict() == {}
    assert id(source.weight) == source_weight_id
    assert source.weight.data_ptr() == source_weight_ptr
    assert torch.equal(source.weight, source_before)

    logits_processor = LogitsProcessor(vocab_size=509)
    logits_processor.head_dtype = torch.bfloat16
    with patch.object(logits_processor, "_gather_logits") as gather:
        processed = logits_processor(proposal, hidden)
    gather.assert_not_called()
    torch.testing.assert_close(processed, actual[..., :509])
