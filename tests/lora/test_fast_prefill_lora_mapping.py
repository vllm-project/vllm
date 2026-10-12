# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.model_executor.models import gemma3n, gemma4


@pytest.mark.skip_global_cleanup
def test_fast_prefill_lora_mapping_rebuilds_compact_metadata():
    wrapper = PunicaWrapperGPU(
        max_num_batched_tokens=16,
        max_batches=2,
        device="cpu",
        lora_config=SimpleNamespace(max_loras=2, specialize_active_lora=False),
    )
    wrapper._token_lora_indices[:6] = torch.tensor([-1, -1, -1, -1, 0, 0])
    wrapper.indices_len[0] = 6

    compact_mapping = wrapper.prepare_fast_prefill_token_mapping(
        torch.tensor([3, 5], dtype=torch.int32)
    )
    assert compact_mapping is not None
    assert compact_mapping.tolist() == [-1, 0]

    (
        token_lora_mapping,
        sorted_token_indices,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        no_lora_flag,
        _,
    ) = wrapper.fast_prefill_mapping_meta.meta_args(2, False)
    assert token_lora_mapping.tolist() == [-1, 0]
    assert sorted_token_indices.tolist() == [0, 1]
    assert num_tokens_per_lora[:3].tolist() == [1, 1, 0]
    assert lora_token_start_loc[:4].tolist() == [0, 1, 2, 0]
    assert active_lora_ids[:2].tolist() == [-1, 0]
    assert not no_lora_flag.item()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("is_fast_prefill", [False, True])
def test_lora_metadata_selection_follows_batch_descriptor(monkeypatch, is_fast_prefill):
    wrapper = PunicaWrapperGPU(
        max_num_batched_tokens=16,
        max_batches=2,
        device="cpu",
        lora_config=SimpleNamespace(max_loras=2, specialize_active_lora=False),
    )
    full_mapping = torch.tensor([-1, -1, 0, 0], dtype=torch.int32)
    wrapper._token_lora_indices[: full_mapping.numel()] = full_mapping
    wrapper.indices_len[0] = full_mapping.numel()
    wrapper.token_mapping_meta.prepare_tensors(full_mapping)
    compact_mapping = wrapper.prepare_fast_prefill_token_mapping(
        torch.tensor([1, 3], dtype=torch.int32)
    )
    assert compact_mapping is not None

    context = SimpleNamespace(
        batch_descriptor=BatchDescriptor(
            num_tokens=2,
            has_lora=True,
            is_fast_prefill=is_fast_prefill,
        ),
        lora_token_mapping=full_mapping,
    )
    monkeypatch.setattr(
        "vllm.lora.punica_wrapper.punica_gpu.is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.lora.punica_wrapper.punica_gpu.get_forward_context",
        lambda: context,
    )

    (token_lora_mapping, *_) = wrapper._token_mapping_meta_args(2)
    expected = (
        wrapper.fast_prefill_mapping_meta
        if is_fast_prefill
        else wrapper.token_mapping_meta
    )
    assert token_lora_mapping.data_ptr() == expected.token_lora_mapping.data_ptr()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_module", [gemma3n, gemma4])
@pytest.mark.parametrize(
    "use_fast_prefill,raise_in_cross_decoder",
    [
        pytest.param(False, False, id="full"),
        pytest.param(True, False, id="compact"),
        pytest.param(True, True, id="compact-error"),
    ],
)
def test_fast_prefill_cross_decoder_restores_forward_context(
    monkeypatch,
    model_module,
    use_fast_prefill,
    raise_in_cross_decoder,
):
    compact_mapping = torch.tensor([-1, 0])
    full_mapping = torch.tensor([-1, -1, 0, 0])
    descriptor = BatchDescriptor(num_tokens=4, has_lora=True, num_active_loras=1)
    context = SimpleNamespace(
        attn_metadata=(
            {
                "last": SimpleNamespace(
                    logits_indices_padded=torch.tensor([1, 3]),
                    num_logits_indices=2,
                )
            }
            if use_fast_prefill
            else None
        ),
        batch_descriptor=descriptor,
        lora_token_mapping=full_mapping,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        skip_compiled=False,
        additional_kwargs={"fast_prefill_lora_mapping": compact_mapping},
    )
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)

    def make_cross_decoder(is_compact):
        def cross_decoder(*args, **kwargs):
            assert is_compact is use_fast_prefill
            expected_mapping = compact_mapping if use_fast_prefill else full_mapping
            assert context.lora_token_mapping is expected_mapping
            assert context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
            assert context.skip_compiled is False
            assert context.batch_descriptor.num_tokens == (2 if use_fast_prefill else 4)
            assert context.batch_descriptor.is_fast_prefill is use_fast_prefill
            if raise_in_cross_decoder:
                raise ValueError("cross decoder error")
            return torch.zeros(2 if use_fast_prefill else 4, 4)

        return cross_decoder

    model = SimpleNamespace(
        positions=torch.zeros(4, dtype=torch.long),
        hidden_states=torch.zeros(4, 4),
        per_layer_inputs=torch.zeros(4, 4),
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name="last"))
            )
        ],
        fast_prefill_enabled=True,
        self_decoder=lambda **kwargs: (torch.zeros(4, 4), torch.zeros(4, 4)),
        cross_decoder=make_cross_decoder(False),
        _fast_prefill_cross_decoder=make_cross_decoder(True),
    )
    model_class = (
        gemma4.Gemma4Model if model_module is gemma4 else gemma3n.Gemma3nTextModel
    )
    if raise_in_cross_decoder:
        with pytest.raises(ValueError, match="cross decoder error"):
            model_class.fast_prefill_forward(
                model, torch.zeros(4, dtype=torch.long), torch.arange(4)
            )
    else:
        model_class.fast_prefill_forward(
            model, torch.zeros(4, dtype=torch.long), torch.arange(4)
        )

    assert context.lora_token_mapping is full_mapping
    assert context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    assert context.skip_compiled is False
    assert context.batch_descriptor is descriptor
