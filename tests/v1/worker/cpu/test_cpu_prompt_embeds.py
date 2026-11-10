# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import LlamaConfig

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.cpu.model_states.default import DefaultModelState
from vllm.v1.worker.cpu.states import CPURequestState

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

HIDDEN_SIZE = 4
VOCAB_SIZE = 32
MAX_NUM_REQS = 4
MAX_NUM_TOKENS = 64
DEVICE = torch.device("cpu")


class _EmbedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor] | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed(input_ids)


@pytest.fixture(params=[False, True], ids=["text-only", "multimodal"])
def state(request, tmp_path) -> DefaultModelState:
    hf_config = LlamaConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=MAX_NUM_TOKENS,
        vocab_size=VOCAB_SIZE,
        architectures=["LlamaForCausalLM"],
    )
    hf_config.save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="float32",
            max_model_len=MAX_NUM_TOKENS,
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=MAX_NUM_TOKENS,
            is_encoder_decoder=False,
            max_num_batched_tokens=MAX_NUM_TOKENS,
            max_num_seqs=MAX_NUM_REQS,
        ),
    )
    # A multimodal model overlays onto the encoder runner's buffer instead.
    encoder_cache = EncoderCache() if request.param else None
    return DefaultModelState(vllm_config, _EmbedModel(), encoder_cache, DEVICE)


@pytest.fixture
def req_states() -> CPURequestState:
    return CPURequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_NUM_TOKENS,
        max_num_batched_tokens=MAX_NUM_TOKENS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )


def _add_request(
    state: DefaultModelState,
    req_states: CPURequestState,
    req_id: str,
    num_computed_tokens: int,
    prompt_embeds: torch.Tensor | None = None,
    prompt_is_token_ids: list[bool] | None = None,
) -> None:
    # Longer than every prompt here, as for a request resumed after preemption.
    token_ids = [0] * 8
    prompt_len = len(token_ids) if prompt_embeds is None else prompt_embeds.shape[0]
    req_states.add_request(
        req_id=req_id,
        prompt_len=prompt_len,
        all_token_ids=token_ids,
        num_computed_tokens=num_computed_tokens,
        max_tokens=1,
    )
    if state.supports_mm_inputs:
        state.encoder_cache.add_request(req_id, [])
    state.add_request(
        req_states.req_id_to_index[req_id],
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=None if prompt_embeds is not None else token_ids,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(),
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
            prefill_token_ids=token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
        ),
    )


def _prepare_inputs_embeds(
    state: DefaultModelState,
    req_states: CPURequestState,
    req_ids: list[str],
    tokens_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one step; return (inputs_embeds, token-id embeddings)."""
    input_batch = InputBatch.make_dummy(
        num_reqs=len(req_ids),
        num_tokens=len(req_ids) * tokens_per_req,
        input_buffers=InputBuffers(MAX_NUM_REQS, MAX_NUM_TOKENS, DEVICE),
        is_padding=False,
    )
    indices = [req_states.req_id_to_index[req_id] for req_id in req_ids]
    input_batch.req_ids[:] = req_ids
    input_batch.idx_mapping.copy_(torch.tensor(indices))
    input_batch.idx_mapping_np[:] = indices
    input_batch.num_computed_tokens_np[:] = req_states.num_computed_tokens_np[indices]
    input_batch.input_ids.copy_(
        torch.randint(VOCAB_SIZE, input_batch.input_ids.shape, dtype=torch.int32)
    )

    inputs_embeds = state.prepare_inputs_embeds({}, input_batch, req_states)
    return inputs_embeds, state.model.embed_input_ids(input_batch.input_ids)


@torch.inference_mode()
def test_overlay_follows_prefill_progress(
    state: DefaultModelState, req_states: CPURequestState
) -> None:
    """Rows come from prompt_embeds at the num_computed offset. Requests
    without embeds, and rows past the embeds length, keep the token-id rows."""
    embeds_a = torch.randn(6, HIDDEN_SIZE)
    embeds_b = torch.randn(5, HIDDEN_SIZE)
    _add_request(state, req_states, "c", num_computed_tokens=1)
    _add_request(state, req_states, "a", num_computed_tokens=2, prompt_embeds=embeds_a)
    _add_request(state, req_states, "b", num_computed_tokens=3, prompt_embeds=embeds_b)

    out, base = _prepare_inputs_embeds(state, req_states, ["a", "b", "c"], 4)

    assert torch.equal(out[:4], embeds_a[2:])
    assert torch.equal(out[4:6], embeds_b[3:])
    assert torch.equal(out[6:], base[6:])


@torch.inference_mode()
def test_overlay_keeps_token_id_rows(
    state: DefaultModelState, req_states: CPURequestState
) -> None:
    """Mixed mode: positions marked as token ids keep the token-id rows."""
    embeds = torch.randn(6, HIDDEN_SIZE)
    is_token_ids = torch.tensor([True, False, False, True, False, True])
    _add_request(
        state,
        req_states,
        "a",
        num_computed_tokens=2,
        prompt_embeds=embeds,
        prompt_is_token_ids=is_token_ids.tolist(),
    )

    out, base = _prepare_inputs_embeds(state, req_states, ["a"], 4)

    assert torch.equal(out, torch.where(is_token_ids[2:, None], base, embeds[2:]))


@torch.inference_mode()
def test_reused_index_does_not_inherit_prompt_embeds(
    state: DefaultModelState, req_states: CPURequestState
) -> None:
    other = torch.randn(2, HIDDEN_SIZE)
    _add_request(
        state,
        req_states,
        "a",
        num_computed_tokens=0,
        prompt_embeds=torch.randn(4, HIDDEN_SIZE),
    )
    state.remove_request("a")
    req_states.remove_request("a")
    _add_request(state, req_states, "b", num_computed_tokens=0)
    _add_request(state, req_states, "other", num_computed_tokens=0, prompt_embeds=other)

    out, base = _prepare_inputs_embeds(state, req_states, ["b", "other"], 2)

    assert torch.equal(out[:2], base[:2])
    assert torch.equal(out[2:], other)
