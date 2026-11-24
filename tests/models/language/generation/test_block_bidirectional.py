# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block attention of the slot decision models against the transformers block
mask, on random four-layer models built from BLOCK_ATTN_BASE and
GEMMA_BLOCK_ATTN_BASE."""

import os
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch

from tests.models.test_gemma4_slot import make_vllm_config, tiny_gemma4_config
from tests.models.test_qwen3_5_slot import OVERRIDES
from tests.v1.core.utils import (
    full_layer,
    new_kv_cache_group,
    new_kv_cache_spec,
    new_sliding_window_layer,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.gemma4_slot import Gemma4SlotDecisionModelState
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.prefix_lm_scheduler import PREFIX_LM_LEN
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

BASE = os.environ.get("BLOCK_ATTN_BASE", "baselabs/elephant-1")
# bison-1 and groundhog-1 share the Gemma 4 processor.
GEMMA_BASE = os.environ.get("GEMMA_BLOCK_ATTN_BASE", "baselabs/bison-1")
NUM_SCORED = 256
# bf16 vLLM against an fp32 reference.
ATOL = 0.1
DEVICE = torch.device("cuda")

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="block attention needs a CUDA GPU"
)


def build_tiny_checkpoint(path: str) -> None:
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(BASE)
    config.text_config.update({
        "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "linear_num_key_heads": 2, "linear_num_value_heads": 2,
        "mtp_num_hidden_layers": 0, "dtype": "float32",
    })  # fmt: skip
    config.vision_config.update({
        "depth": 1, "hidden_size": 64, "num_heads": 2, "intermediate_size": 128,
        "out_hidden_size": 256,
    })  # fmt: skip
    config.modeling_overrides = OVERRIDES
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.float32)
    # Sharper weights and stronger attention values/outputs make the mask move
    # the last token's logits well above bf16 noise.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1:
                strong = "self_attn.v_proj" in name or "self_attn.o_proj" in name
                param.normal_(0.0, 0.32 if strong else 0.08)
    model.save_pretrained(path)
    AutoProcessor.from_pretrained(BASE).save_pretrained(path)


def hf_logprobs(path: str, prompts: dict, prefix_len: int) -> list[dict]:
    """Last-position logprobs over the first NUM_SCORED tokens of each prompt,
    with the prefix and the rest as two blocks, then causal."""
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda")
    mask = hf.create_causal_mask

    def run(prompt: list[int], blocked: bool) -> torch.Tensor:
        ids = (torch.arange(len(prompt), device="cuda") >= prefix_len).long()[None]
        patched = partial(mask, block_sequence_ids=ids if blocked else None)
        monkeypatch.setattr(hf, "create_causal_mask", patched)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([prompt], device="cuda")).logits
        return torch.log_softmax(logits[0, -1].float(), -1)[:NUM_SCORED].cpu()

    with pytest.MonkeyPatch.context() as monkeypatch:
        return [{q: run(p, b) for q, p in prompts.items()} for b in (True, False)]


def build_tiny_gemma4_checkpoint(
    path: str, moe: bool = False, head_dims: tuple[int, int] = (64, 128)
) -> None:
    """A random four-layer Gemma4ForSlotDecision, dense or MoE."""
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    torch.manual_seed(0)
    config = tiny_gemma4_config(moe, head_dims=head_dims)
    model = Gemma4ForConditionalGeneration(config).to(torch.float32)
    # As for Qwen; the full layers have K = V, so no v_proj.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() > 1:
                strong = "self_attn.v_proj" in name or "self_attn.o_proj" in name
                param.normal_(0.0, 0.32 if strong else 0.08)
    model.save_pretrained(path)
    AutoProcessor.from_pretrained(GEMMA_BASE).save_pretrained(path)


def gemma4_hf_logprobs(path: str, prompts: dict, prefix_len: int) -> list[dict]:
    """As hf_logprobs, with causal sliding layers; also returns the logprobs with
    the blocks on the sliding layers, which vLLM must not match."""
    from transformers import Gemma4ForConditionalGeneration
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    model = Gemma4ForConditionalGeneration.from_pretrained(
        path, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda")
    text_config = model.config.get_text_config()

    def run(prompt: list[int], full: bool, sliding: bool) -> torch.Tensor:
        input_ids = torch.tensor([prompt], device="cuda")
        ids = (torch.arange(len(prompt), device="cuda") >= prefix_len).long()[None]
        mask_kwargs = {
            "config": text_config,
            "inputs_embeds": model.get_input_embeddings()(input_ids),
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": torch.arange(len(prompt), device="cuda")[None],
        }
        # A prepared mask per layer type, which the model uses as it is.
        masks = {
            "full_attention": create_causal_mask(
                **mask_kwargs, block_sequence_ids=ids if full else None
            ),
            "sliding_attention": create_sliding_window_causal_mask(
                **mask_kwargs, block_sequence_ids=ids if sliding else None
            ),
        }
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=masks).logits
        return torch.log_softmax(logits[0, -1].float(), -1)[:NUM_SCORED].cpu()

    layouts = [(True, False), (False, False), (True, True)]
    return [{q: run(p, *layout) for q, p in prompts.items()} for layout in layouts]


@pytest.mark.hybrid_model
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_block_attention_matches_transformers(vllm_runner, tmp_path, enforce_eager):
    check_block_attention(
        vllm_runner, str(tmp_path), enforce_eager, build_tiny_checkpoint, hf_logprobs
    )


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_gemma4_block_attention_matches_transformers(
    vllm_runner, tmp_path, enforce_eager
):
    check_block_attention(
        vllm_runner,
        str(tmp_path),
        enforce_eager,
        build_tiny_gemma4_checkpoint,
        gemma4_hf_logprobs,
        # This checkpoint amplifies bf16 rounding.
        dtype="float32",
    )


# Where the full layers run the PrefixLM blocks as varlen segments.
SEGMENTS = (
    AttentionBackendEnum.FLASH_ATTN
    if current_platform.is_device_capability_family(90)
    else AttentionBackendEnum.FLASHINFER
    if current_platform.is_device_capability_family(100)
    else AttentionBackendEnum.TRITON_ATTN
)


@pytest.mark.parametrize(
    "engine_args,backend",
    [
        ({}, SEGMENTS),
        ({"attention_backend": "TRITON_ATTN"}, AttentionBackendEnum.TRITON_ATTN),
        # FlashAttention and FlashInfer run fp16 and bf16 only.
        ({"dtype": "float32"}, AttentionBackendEnum.TRITON_ATTN),
        # FA4 non-causal attention on an SM90 FP8 KV cache is inaccurate.
        ({"kv_cache_dtype": "fp8"}, AttentionBackendEnum.TRITON_ATTN),
    ],
    ids=["default", "triton", "fp32", "fp8_kv_cache"],
)
def test_gemma4_full_attention_backend(tmp_path, engine_args, backend):
    config = make_vllm_config(tmp_path, head_dims=(256, 512), **engine_args)
    attention_config = config.attention_config
    full = attention_config.backend_per_kind.get(
        "full_attention", attention_config.backend
    )
    assert full == backend


@pytest.mark.skipif(
    SEGMENTS == AttentionBackendEnum.TRITON_ATTN,
    reason="PrefixLM blocks run as varlen segments on SM90 and SM100 only",
)
def test_gemma4_segments_block_attention_matches_transformers(vllm_runner, tmp_path):
    """As above, with the checkpoints' head sizes and the full layers on varlen
    segments, held to Triton in fp16, whose rounding moves the reads past ATOL."""
    checkpoint = str(tmp_path)
    build = partial(build_tiny_gemma4_checkpoint, head_dims=(256, 512))
    prefix, questions, expected, wrong = block_attention_references(
        checkpoint, build, gemma4_hf_logprobs
    )
    reads = partial(vllm_reads, vllm_runner, checkpoint, False, prefix, questions)
    segments = reads("float16")
    triton = reads("float16", attention_backend="TRITON_ATTN")
    for name, got in segments.items():
        reference = reference_of(name, expected, wrong[0])
        error = (got - reference).abs().max().item()
        triton_error = (triton[name] - reference).abs().max().item()
        assert error < triton_error + ATOL, (name, error, triton_error)
        if name != "q0 causal":
            for other in wrong:
                q = name.split()[0]
                assert error + 5 * ATOL < (got - other[q]).abs().max().item(), name


def check_block_attention(
    vllm_runner, checkpoint: str, enforce_eager, build, reference, dtype="bfloat16"
):
    prefix, questions, expected, wrong = block_attention_references(
        checkpoint, build, reference
    )
    reads = vllm_reads(vllm_runner, checkpoint, enforce_eager, prefix, questions, dtype)
    for name, got in reads.items():
        error = (got - reference_of(name, expected, wrong[0])).abs().max().item()
        assert error < ATOL, (name, error)


def reference_of(name: str, expected: dict, causal: dict) -> torch.Tensor:
    """The reference of a read named by vllm_reads."""
    return causal["q0"] if name == "q0 causal" else expected[name.split()[0]]


def block_attention_references(checkpoint: str, build, reference):
    """The prompts and their references, block-masked first, which the other
    masks must move."""
    build(checkpoint)
    g = torch.Generator().manual_seed(1)

    def tokens(n: int) -> list[int]:
        return torch.randint(1000, 100000, (n,), generator=g).tolist()

    # A long shared prefix, then three questions.
    prefix = tokens(700)
    shared = tokens(40)
    questions = {
        "q0": prefix + tokens(90),
        "q1": prefix + shared + tokens(50),
        "q2": prefix + shared + tokens(50),
    }
    expected, *wrong = reference(checkpoint, questions, len(prefix))
    torch.accelerator.empty_cache()
    for q in questions:
        # The mask must matter, or the comparison proves nothing.
        for other in wrong:
            assert (other[q] - expected[q]).abs().max() > 5 * ATOL, q
    return prefix, questions, expected, wrong


def vllm_reads(
    vllm_runner, checkpoint, enforce_eager, prefix, questions, dtype, **engine_args
) -> dict[str, torch.Tensor]:
    """The reads of a vLLM run by name: each question, "q0 causal" and, with
    prefix caching, "q1 cached" and so on."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    p = len(prefix)

    def params(prefix_len: int | None) -> SamplingParams:
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprob_token_ids=list(range(NUM_SCORED)),
            extra_args=None if prefix_len is None else {"prefix_lm_len": prefix_len},
        )

    def logprobs(out) -> torch.Tensor:
        lp = out.outputs[0].logprobs[0]
        return torch.tensor([lp[t].logprob for t in range(NUM_SCORED)])

    # Room for every prompt in one step, so the reads share a batch.
    with vllm_runner(
        checkpoint,
        dtype=dtype,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={"image": 0, "video": 0},
        **engine_args,
    ) as runner:
        llm = runner.llm
        vllm_config = llm.llm_engine.vllm_config
        assert vllm_config.model_config.head_dtype == torch.float32
        # Prefix caching stays on with FlashInfer's GDN prefill only.
        prefix_caching = vllm_config.cache_config.enable_prefix_caching
        assert prefix_caching or vllm_config.model_config.is_hybrid
        if prefix_caching:
            scheduler_cls = vllm_config.scheduler_config.scheduler_cls
            assert scheduler_cls.startswith("vllm.v1.core.sched.prefix_lm_scheduler.")

        # The three questions with their blocks, then q0 without a prefix length,
        # which is read causally.
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=questions[q]) for q in [*questions, "q0"]],
            [params(p)] * len(questions) + [params(None)],
            use_tqdm=False,
        )
        reads = {
            name: logprobs(out) for name, out in zip([*questions, "q0 causal"], outputs)
        }

        if not prefix_caching:
            return reads
        # The prefix alone, as the strategy's warm-up sends it. The questions
        # then reuse its cache, and never past it: q1 and q2 share 40 more
        # tokens, and q2 comes twice.
        llm.generate(TokensPrompt(prompt_token_ids=prefix), params(p), use_tqdm=False)
        for name in ("q1 cached", "q2 cached", "q2 cached again"):
            (out,) = llm.generate(
                TokensPrompt(prompt_token_ids=questions[name.split()[0]]),
                params(p),
                use_tqdm=False,
            )
            reads[name] = logprobs(out)
            assert 0 < (out.num_cached_tokens or 0) <= p, out.num_cached_tokens
    return reads


SLIDING, FULL = "model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"
#: Both kinds of group hold an AttentionSpec.
KV_CACHE_CONFIG = KVCacheConfig(
    num_blocks=8,
    kv_cache_tensors=[],
    kv_cache_groups=[
        new_kv_cache_group(
            [SLIDING], new_sliding_window_layer(sliding_window=128, head_size=64)
        ),
        new_kv_cache_group(
            [FULL], full_layer(new_kv_cache_spec(head_size=128, num_kv_heads=1))
        ),
    ],
)


def new_request(req_id, prompt_len, prefix_len=None, image=False):
    ids = list(range(prompt_len))
    image_feature = MultiModalFeatureSpec(
        None, "image", req_id, PlaceholderRange(offset=2, length=4)
    )
    extra_args = None if prefix_len is None else {PREFIX_LM_LEN: prefix_len}
    return NewRequestData(
        req_id,
        ids,
        [image_feature] if image else [],
        SamplingParams(extra_args=extra_args),
        None,
        ([], []),
        0,
        None,
        prefill_token_ids=ids,
    )


def new_input_batch(req_ids, prefilling, scheduled, computed) -> InputBatch:
    n, num_tokens = len(req_ids), sum(scheduled)
    batch = InputBatch.make_dummy(n, num_tokens, InputBuffers(n, num_tokens, DEVICE))
    query_start_loc = np.concatenate([[0], np.cumsum(scheduled)]).astype(np.int32)
    seq_lens = torch.tensor(computed, dtype=torch.int32) + torch.tensor(scheduled)
    return replace(
        batch,
        req_ids=req_ids,
        num_scheduled_tokens=np.array(scheduled, dtype=np.int32),
        query_start_loc=torch.from_numpy(query_start_loc),
        query_start_loc_np=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        num_computed_prefill_tokens_np=np.array(computed, dtype=np.int32),
        is_prefilling_np=np.array(prefilling),
    )


@pytest.mark.parametrize(
    "req_ids,prefilling,scheduled,computed,mode,full,sliding",
    [
        # PrefixLM, image and causal prefills: the blocks reach the full layers
        # only, and the image ranges the sliding layers only.
        (
            ["plm", "img", "causal"],
            [True] * 3,
            [10, 12, 4],
            [0, 0, 0],
            CUDAGraphMode.PIECEWISE,
            {0: [(0, 5), (6, 9)]},
            {0: [], 1: [(2, 5)], 2: []},
        ),
        # Without a PrefixLM request the full layers stay causal.
        (
            ["img", "causal"],
            [True, True],
            [12, 4],
            [0, 0],
            CUDAGraphMode.PIECEWISE,
            None,
            {0: [(2, 5)], 1: []},
        ),
        # A FULL graph may run a PrefixLM prompt's last token, without ranges.
        (
            ["plm"],
            [True],
            [1],
            [9],
            CUDAGraphMode.FULL,
            {0: []},
            {0: []},
        ),
    ],
)
def test_prepare_attn_gives_the_blocks_to_full_attention_groups_only(
    tmp_path, req_ids, prefilling, scheduled, computed, mode, full, sliding
):
    """A PrefixLM, an image and a causal request; checks each layer's ranges."""
    config = make_vllm_config(tmp_path)
    state = Gemma4SlotDecisionModelState(config, torch.nn.Module(), None, DEVICE)
    groups = KV_CACHE_CONFIG.kv_cache_groups
    block_tables = BlockTables(
        block_sizes=[group.block_size for group in groups],
        max_num_reqs=4,
        max_num_batched_tokens=32,
        max_num_blocks_per_group=[8] * len(groups),
        device=DEVICE,
    )
    state.initialize_kv_cache(KV_CACHE_CONFIG, block_tables)
    attn_groups = []
    for i, group in enumerate(groups):
        (name, spec), *_ = group.kv_cache_specs.items()
        attn_group = AttentionGroup(TritonAttentionBackend, [name], spec, i)
        attn_group.create_metadata_builders(config, DEVICE, kv_cache_group=group)
        attn_groups.append([attn_group])
    state.add_request(0, new_request("plm", 10, prefix_len=6))
    state.add_request(1, new_request("img", 12, image=True))
    state.add_request(2, new_request("causal", 4))

    batch = new_input_batch(req_ids, prefilling, scheduled, computed)
    metadata = state.prepare_attn(
        batch,
        mode,
        block_tables.get_dummy_block_tables(batch.num_reqs),
        block_tables.get_dummy_slot_mappings(batch.num_tokens),
        attn_groups,
        KV_CACHE_CONFIG,
    )
    assert metadata[FULL].mm_prefix_range == full
    assert metadata[SLIDING].mm_prefix_range == sliding
