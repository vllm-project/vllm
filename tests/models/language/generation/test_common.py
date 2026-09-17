# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.platforms import current_platform

from ....utils import large_gpu_mark
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

# Models that require embedding scaling for prompt_embeds test
EMBED_SCALING_MODELS = {
    "openbmb/MiniCPM4.1-8B",
}

# This list contains the model that are using AITER kernel.
# Skip model that are not using AITER tests.
# When more AITER kernels are added, this list will not be
# needed as all the models will be calling AITER kernels
# in parts of the operators
AITER_MODEL_LIST = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "openbmb/MiniCPM3-4B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TitanML/tiny-mixtral",
    "Qwen/Qwen3-8B",
]

# Largest gap tolerated between HF's and vLLM's router logits, in fp32, reading
# the same tokens. Measured on tiny-mixtral: 0.015625 over all 856 rows, one
# bf16 ULP at that magnitude, which is the whole of the drift and is enough to
# decide an otherwise exact top-k tie. Set well above that so a last-bit
# difference is not news, and far below the ~1.0 that two routers reading
# different hidden states drift apart by.
MOE_ROUTER_LOGIT_TOL = 0.05


@contextmanager
def record_hf_routing(
    hf_model,
    hf_logits: dict[int, torch.Tensor],
    hf_ids: dict[int, torch.Tensor],
):
    """Record HF's router logits and top-k expert ids, per MoE layer.

    Rows accumulate in call order, which is the order vLLM's routers see the
    same tokens in. Both stores stay empty for a dense model, so the caller can
    tell one apart without asking the config.
    """
    # A router is a `gate` that owns `experts`; a dense gated MLP has neither.
    # MoE layers only, so the keys match the order vLLM first sees its routers in.
    gates = [
        layer.mlp.gate
        for layer in getattr(getattr(hf_model.model, "model", None), "layers", [])
        if hasattr(getattr(layer, "mlp", None), "gate")
        and hasattr(layer.mlp, "experts")
    ]
    if not gates:
        yield
        return

    logit_rows: dict[int, list[torch.Tensor]] = {}
    id_rows: dict[int, list[torch.Tensor]] = {}

    def record(module, args, out, i):
        # A router returns (router_logits, router_scores, router_indices);
        # anything else is not one. Transformers reads index 0 the same way,
        # via OutputRecorder(MixtralTopKRouter, index=0).
        if isinstance(out, tuple) and len(out) == 3:
            for store, tensor in ((logit_rows, out[0]), (id_rows, out[2])):
                store.setdefault(i, []).append(
                    tensor.detach().reshape(-1, tensor.shape[-1]).cpu()
                )

    hooks = [
        gate.register_forward_hook(lambda m, args, out, i=i: record(m, args, out, i))
        for i, gate in enumerate(gates)
    ]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
        hf_logits.update({i: torch.cat(v) for i, v in logit_rows.items()})
        hf_ids.update({i: torch.cat(v) for i, v in id_rows.items()})


@contextmanager
def patched_select_experts(replacement):
    """Run `replacement` in place of every MoE router's `select_experts`.

    `FusedMoERouter.select_experts` is the one place where a row's router logits
    and the experts chosen from them are both in hand, which is what both the
    comparison and the substitution below need.
    """
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )

    original = FusedMoERouter.select_experts
    layer_of: dict[int, int] = {}
    cursor: dict[int, int] = {}

    def select_experts(
        self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None
    ):
        weights, ids = original(
            self, hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
        )
        # Routers run in layer order, so first-seen order is layer order.
        layer = layer_of.setdefault(id(self), len(layer_of))
        start = cursor.get(layer, 0)
        cursor[layer] = start + ids.shape[0]
        return replacement(self, layer, start, router_logits, weights, ids)

    with patch.object(FusedMoERouter, "select_experts", select_experts):
        yield


@contextmanager
def compare_hf_routing(
    hf_logits: dict[int, torch.Tensor],
    hf_ids: dict[int, torch.Tensor],
):
    """Measure how far vLLM's router logits sit from the ones HF recorded.

    Rows are matched against `record_hf_routing`'s by position, which only means
    anything while both sides are looking at the same token. Generating freely
    does not give that: bf16 drift anywhere in the model eventually picks a
    different greedy token, and from there the two are reading different text.
    So the caller has to hand vLLM the sequence HF actually produced and let it
    prefill, under eager execution -- graph replay skips Python entirely.

    Yields a report of the rows compared per layer, the largest absolute gap
    between the two sides' logits, and every row where they chose differently.
    """
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    report: dict[str, Any] = {
        "rows": {},
        "max_logit_delta": 0.0,
        "worst": None,
        "disagreements": [],
    }

    def compare(router, layer, start, router_logits, weights, ids):
        n = ids.shape[0]
        their_ids = hf_ids.get(layer, ids.new_empty(0))[start : start + n]
        their_logits = hf_logits.get(layer, ids.new_empty(0))[start : start + n]
        if their_ids.shape[0] != n or their_logits.shape[0] != n:
            # HF has run out of rows. Let the caller's row count come up short
            # rather than compare rows that are not the same tokens.
            return weights, ids

        with gpu_sync_allowed():  # HF's rows have to come back to the device
            mine = router_logits.float()
            theirs = their_logits.to(mine.device).float()
            delta = (theirs - mine).abs()
            flat = delta.flatten()
            at = int(flat.argmax())
            biggest = float(flat[at])
            if biggest > report["max_logit_delta"]:
                row, expert = divmod(at, delta.shape[-1])
                report["max_logit_delta"] = biggest
                report["worst"] = (
                    f"layer {layer} row {start + row} expert {expert}: "
                    f"hf {float(theirs[row, expert])!r} vs "
                    f"vllm {float(mine[row, expert])!r}"
                )

            # Order carries no meaning here, only the pair that was chosen.
            their_ids_d = their_ids.to(ids.device).long()
            differs = (ids.long().sort(-1).values != their_ids_d.sort(-1).values).any(-1)
            for row in differs.nonzero().flatten().tolist():
                report["disagreements"].append(
                    f"layer {layer} row {start + row}: "
                    f"hf {their_ids_d[row].tolist()} vs vllm {ids[row].tolist()}, "
                    f"logits apart by {float(delta[row].max()):.6f}"
                )
            report["rows"][layer] = start + n
        return weights, ids

    with patched_select_experts(compare):
        yield report


@contextmanager
def use_hf_routing(hf_ids: dict[int, torch.Tensor]):
    """Send each token to the experts HF sent it to, keeping vLLM's weights.

    Only the choice comes from HF; the weights handed back are rebuilt from
    vLLM's own logits, the way the router would have built them.

    Rows are matched by position, so enter once per generate call under eager
    execution. Once a sequence's greedy tokens diverge from HF's the rows stop
    describing the same token, but `check_logprobs_close` stops comparing a
    sequence at its first differing token, so only the rows that still line up
    can affect its verdict.
    """
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    def force(router, layer, start, router_logits, weights, ids):
        n = ids.shape[0]
        their_ids = hf_ids.get(layer, ids.new_empty(0))[start : start + n]
        if their_ids.shape[0] != n:
            return weights, ids  # HF has run out of rows; leave vLLM alone.

        with gpu_sync_allowed():  # HF's rows have to come back to the device
            their_ids = their_ids.to(ids.device).long()
            forced = router_logits.float().softmax(-1).gather(1, their_ids)
            if getattr(router, "renormalize", True):
                forced = forced / forced.sum(-1, keepdim=True)
            order = forced.argsort(-1, descending=True)
            return (
                forced.gather(1, order).to(weights.dtype),
                their_ids.gather(1, order).to(ids.dtype),
            )

    with patched_select_experts(force):
        yield


# @maybe_test_rocm_aiter
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "bigscience/bloom-560m",  # bloom - testing alibi slopes
            marks=[
                pytest.mark.core_model,
                pytest.mark.slow_test,
                pytest.mark.cpu_model,
            ],
        ),
        pytest.param(
            "openai-community/gpt2",  # gpt2
            marks=[pytest.mark.core_model],
        ),
        pytest.param("Milos/slovak-gpt-j-405M"),  # gptj
        pytest.param("bigcode/tiny_starcoder_py"),  # gpt_bigcode
        pytest.param("EleutherAI/pythia-70m"),  # gpt_neox
        pytest.param(
            "google/gemma-1.1-2b-it",  # gemma
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param(
            "google/gemma-2-2b-it",  # test hybrid attention
            marks=[pytest.mark.cpu_model],
        ),
        pytest.param(
            "zai-org/chatglm3-6b",  # chatglm (text-only)
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",  # llama
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openbmb/MiniCPM4.1-8B",  # minicpm
            marks=[pytest.mark.core_model, large_gpu_mark(min_gb=48)],
        ),
        pytest.param(
            "facebook/opt-125m",  # opt
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "microsoft/phi-2",  # phi
            marks=[pytest.mark.core_model, pytest.mark.slow_test],
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param(
            "Qwen/Qwen3-8B",  # qwen (text-only)
        ),
        pytest.param("stabilityai/stablelm-3b-4e1t"),  # stablelm
        pytest.param("bigcode/starcoder2-3b"),  # starcoder2
        pytest.param(
            "TitanML/tiny-mixtral",  # mixtral
            marks=[pytest.mark.core_model],
        ),
        pytest.param("swiss-ai/Apertus-8B-Instruct-2509"),  # apertus
        pytest.param(
            "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",  # hyperclovax
            marks=[large_gpu_mark(min_gb=32)],
        ),
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
@pytest.mark.parametrize("use_prompt_embeds", [True, False])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
    use_rocm_aiter: bool,
    use_prompt_embeds: bool,
    monkeypatch,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    if current_platform.is_rocm() and model == "TitanML/tiny-mixtral":
        # Its single-token router selects LLMM1, whose low-precision
        # accumulation can change the top-2 experts. Keep the optimized kernel
        # enabled generally, but use the reference GEMM for this accuracy test.
        monkeypatch.setenv("VLLM_ROCM_USE_SKINNY_GEMM", "0")

    if use_rocm_aiter and (model in AITER_MODEL_LIST):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    elif use_rocm_aiter and model not in AITER_MODEL_LIST:
        # Skip model that are not using AITER tests.
        # When more AITER kernels are added, this list will not be
        # needed as all the models will be calling AITER kernels
        # in parts of the operators
        pytest.skip(f"Skipping '{model}' model test with AITER kernel.")

    if model == "bigcode/starcoder2-3b":
        # Replace example.txt's Test1 (an NL prompt) with a code prompt:
        # starcoder2-3b is a code model, so NL prompts give near-uniform
        # digit logits where HF<->vLLM bf16 drift can reorder top-K.
        example_prompts = list(example_prompts)
        example_prompts[1] = (
            "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - "
        )

    # Filled in only for a MoE model, which is the one case where a logprob
    # difference can come from the router rather than from the model.
    hf_logits: dict[int, torch.Tensor] = {}
    hf_ids: dict[int, torch.Tensor] = {}

    with hf_runner(
        model,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
    ) as hf_model:
        with record_hf_routing(hf_model, hf_logits, hf_ids):
            hf_outputs = hf_model.generate_greedy_logprobs_limit(
                example_prompts, max_tokens, num_logprobs
            )

        prompt_embeds: list[torch.Tensor] | None = [] if use_prompt_embeds else None
        prompt_token_ids: list[list[int]] = []

        for prompt in example_prompts:
            token_ids = hf_model.tokenizer(prompt, return_tensors="pt").input_ids.to(
                hf_model.model.device
            )
            prompt_token_ids.append(token_ids[0].tolist())
            if prompt_embeds is not None:
                embed = hf_model.model.get_input_embeddings()(token_ids)

                if "gemma" in model.lower() and (
                    Version(TRANSFORMERS_VERSION) < Version("5.3.0.dev0")
                ):
                    # For Gemma 1/2 models with Transformers 5.4.0+, the prompt
                    # embeddings are normalised in `get_prompt_embeddings`,
                    # like Gemma 3. For older versions, we need to manually normalise.
                    embed_scale = hf_model.config.hidden_size**0.5
                    normalizer = torch.tensor(embed_scale, dtype=embed.dtype)
                    embed *= normalizer

                # MiniCPM models apply scale_emb to embeddings internally.
                # vLLM expects pre-scaled embeddings when using inputs_embeds.
                if model in EMBED_SCALING_MODELS:
                    config = hf_model.model.config
                    embed = embed * config.scale_emb

                prompt_embeds.append(embed.squeeze(0))

    vllm_kwargs: dict[str, Any] = {}
    if (
        model == "bigscience/bloom-560m"
        and current_platform.is_device_capability_family(90)
    ):
        # On SM90, the metadata builder otherwise selects FA3 AOT scheduling
        # before Bloom's ALiBi layers fall back to FA2. Pinning FA2 keeps the
        # builder and layer consistent and preserves the L4 test path.
        vllm_kwargs["attention_config"] = {"flash_attn_version": 2}

    if hf_ids:
        # A MoE model may need the routing checks below, and they patch the
        # router from this process: graph replay would skip Python entirely, a
        # spawned EngineCore would never see the patch, and a prefix-cache hit
        # would skip prefill rows that HF's recording still counts.
        vllm_kwargs["enforce_eager"] = True
        vllm_kwargs["enable_prefix_caching"] = False
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with vllm_runner(
        model,
        tokenizer_name=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        # Remove the effects of batch variance on ROCm since batch invariance
        # is not yet supported.
        # See: https://github.com/vllm-project/vllm/issues/27433
        max_num_seqs=1 if current_platform.is_rocm() else 2,
        enable_prompt_embeds=use_prompt_embeds,
        compilation_config={"cudagraph_capture_sizes": [1, 2]},
        **vllm_kwargs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )
        if prompt_embeds is not None:
            vllm_outputs_from_embeds = vllm_model.generate_greedy_logprobs(
                prompt_embeds, max_tokens, num_logprobs
            )

        try:
            check_logprobs_close(
                outputs_0_lst=hf_outputs,
                outputs_1_lst=vllm_outputs,
                name_0="hf",
                name_1="vllm",
            )
        except AssertionError:
            if not hf_ids:
                # Dense model: there is no routing to explain the difference.
                raise

            # A bf16 ULP of drift upstream of the router is enough to decide an
            # otherwise exact top-k tie, and one flipped expert changes every
            # token after it. So ask the two questions the comparison above
            # conflates: do the routers agree, and does the rest of the model
            # agree once it is given the same experts?

            # Prefilling HF's own token streams puts vLLM's routers on exactly
            # the rows HF recorded, which is what makes a row-by-row comparison
            # mean anything: each prompt, then each token HF went on to generate
            # except the last, which nothing was conditioned on. One token is
            # generated because a request has to produce something; it comes
            # after every row being compared.
            hf_token_streams = [
                (prompt + list(output_ids))[:-1]
                for prompt, (output_ids, _, _) in zip(prompt_token_ids, hf_outputs)
            ]
            with compare_hf_routing(hf_logits, hf_ids) as report:
                vllm_model.generate_greedy_logprobs(hf_token_streams, 1, num_logprobs)

            # Every row HF recorded has to have been compared, or this measures
            # nothing: a router that was never patched, or a run that stopped
            # short, would otherwise look like agreement.
            expected_rows = {layer: rows.shape[0] for layer, rows in hf_ids.items()}
            assert report["rows"] == expected_rows, (
                f"compared {report['rows']} router rows, expected {expected_rows}"
            )
            assert report["max_logit_delta"] <= MOE_ROUTER_LOGIT_TOL, (
                f"hf and vllm router logits differ by {report['max_logit_delta']}, "
                f"more than {MOE_ROUTER_LOGIT_TOL}, at {report['worst']}. "
                f"expert choices differed on {len(report['disagreements'])} row(s): "
                f"{report['disagreements'][:10]}"
            )

            # The routers agree, so the tie-break is the only thing between the
            # two runs. Hand vLLM HF's experts and the logprobs have to match.
            with use_hf_routing(hf_ids):
                vllm_outputs_hf_routed = vllm_model.generate_greedy_logprobs(
                    example_prompts, max_tokens, num_logprobs
                )
            check_logprobs_close(
                outputs_0_lst=hf_outputs,
                outputs_1_lst=vllm_outputs_hf_routed,
                name_0="hf",
                name_1="vllm (routed by hf)",
            )

    if prompt_embeds is not None:
        check_logprobs_close(
            outputs_0_lst=vllm_outputs,
            outputs_1_lst=vllm_outputs_from_embeds,
            name_0="vllm",
            name_1="vllm_from_embeds",
        )

    if use_rocm_aiter:
        # this is to ensure that vllm engine
        # has deallocated the memory before running the next
        # unit tests. On ROCm, when using AITER
        # the memory might not be deallocated completely
        # before running the next test case
        torch.accelerator.synchronize()
