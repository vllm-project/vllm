# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Each text-generation architecture that declares `SupportsPP` gives the same
prompt logprobs and greedy tokens with pipeline parallelism as without it.

The registry example of each architecture is cut to a few layers and loaded
with random weights seeded by each parameter's name. Parameter names carry
global layer indices, so every pipeline layout holds the same weights, and no
checkpoint is downloaded.
"""

import hashlib
import math
from functools import partial

import numpy as np
import pytest
import torch

from vllm import ModelRegistry, SamplingParams
from vllm.config.load import LoadConfig
from vllm.inputs import TokensPrompt
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

from ..utils import multi_gpu_test
from .registry import _TEXT_GENERATION_EXAMPLE_MODELS, HF_EXAMPLE_MODELS
from .utils import dummy_hf_overrides, skip_unavailable_models

# Layers per pipeline size: PP=4 has two middle stages, which receive and send.
NUM_LAYERS = {2: 4, 4: 8}
LOAD_FORMAT = "pp_equivalence_random"
PROMPT_LENS = (8, 40, 100)
MAX_TOKENS = 16
# A failing worker can hang the engine instead of failing it.
CASE_TIMEOUT_S = 900

# Run at PP=2 on every PR, where multi-GPU steps have 2 GPUs. Nightly runs every
# architecture at PP=4, whose stages and cuts cover PP=2's.
MINIMAL_PP_ARCH_LIST = [
    "Qwen3ForCausalLM",  # Dense
    "GptOssForCausalLM",  # MoE, sliding window, attention sinks
    "Gemma4ForCausalLM",  # Per-layer inputs, cross-layer KV sharing
    "Qwen3NextForCausalLM",  # Hybrid with linear attention
]

# Architectures that fail on main. A fix removes its entry; `strict` fails the
# test once the fix lands.
# TODO(prototype): link the fix PRs or issues marked TBD.
KNOWN_PP_DEFECTS: dict[str, str] = {
    "Ernie4_5_MoeForCausalLM": "No embedding to tie on the last stage; fixed by #65983",
    "Glm5NextForCausalLM": "Declares PP without implementing it; TBD",
    "GPTBigCodeForCausalLM": "Later stages add position embeddings again; TBD",
    "GraniteMoeHybridForCausalLM": "Loads other stages' layers; fixed by #66007",
    "LongcatFlashNgramForCausalLM": "Later stages build n-gram embeddings; TBD",
}
# The reduced configs keep each model's leading dense layers, which can fill a
# whole stage; these models expect an MoE layer on every stage, as real configs
# have at practical PP sizes.
MOE_ON_EVERY_STAGE: dict[str, dict[str, object]] = {
    "AfmoeForCausalLM": {"num_dense_layers": 0},
    "Lfm2MoeForCausalLM": {"num_dense_layers": 0},
    "Step3p5ForCausalLM": {"mlp_layer_types": ["sparse"] * max(NUM_LAYERS.values())},
}

# Architectures whose layers read the KV cache of an earlier layer, which must be
# on the same stage: the layer count and a `VLLM_PP_LAYER_PARTITION` that puts a
# reader on another stage than its source. Even splits keep them together.
SPLIT_KV_SHARING: dict[str, tuple[int, str]] = {
    # The reduced config shares the last layer's KV cache with the one before.
    "Gemma4ForCausalLM": (4, "3,1"),
}

ModelRegistry.fill_model_info_cache()
# Examples that need remote code are left out until a CI run shows their status.
PP_ARCHS = sorted(
    arch
    for arch, example in _TEXT_GENERATION_EXAMPLE_MODELS.items()
    if not example.trust_remote_code
    and (info := ModelRegistry._try_inspect_model_cls(arch))
    and info.supports_pp
)


@register_model_loader(LOAD_FORMAT)
class NameSeededWeightLoader(DummyModelLoader):
    """Dummy loading, then every floating-point parameter re-seeded from its
    name at a realistic scale, before post-load processing so that derived
    weights follow. `model_loader_extra_config={"perturb_layer": i}` seeds one
    weight of decoder layer `i` differently, for the negative control."""

    def __init__(self, load_config: LoadConfig):
        # DummyModelLoader rejects any extra config; this one reads it.
        BaseModelLoader.__init__(self, load_config)
        extra_config = load_config.model_loader_extra_config or {}
        self.perturb_layer = extra_config.get("perturb_layer")

    @torch.no_grad()
    def load_weights(self, model, model_config) -> None:
        super().load_weights(model, model_config)
        perturbed = False
        # Tied weights appear once, under the same name on every stage.
        for name, param in model.named_parameters():
            if (
                param.device.type == "meta"
                or not torch.is_floating_point(param)
                or param.element_size() < 2
            ):
                continue
            digest = hashlib.sha256(name.encode()).digest()
            seed = int.from_bytes(digest[:8], "little") >> 1
            if (
                not perturbed
                and self.perturb_layer is not None
                and f"layers.{self.perturb_layer}." in name
                and param.dim() == 2
            ):
                seed ^= 0x5BD1E995
                perturbed = True
            generator = torch.Generator(param.device).manual_seed(seed)
            values = torch.randn(param.shape, generator=generator, device=param.device)
            if name.endswith(".A"):
                # Mamba layers hold A = -exp(A_log), which must be negative.
                values = -torch.exp(0.5 * values)
            elif param.dim() >= 2:
                values /= math.sqrt(param.shape[-1])
            else:
                values = 1.0 + 0.1 * values if "norm" in name else 0.02 * values
            param.copy_(values)


class LoaderInWorkers:
    """Worker extension (``worker_extension_cls``): it brings this module, and
    so the load format, into the workers."""


def _run(
    vllm_runner,
    arch: str,
    pp_size: int,
    num_layers: int,
    *,
    perturb_layer: int | None = None,
) -> list[dict]:
    info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    overrides = {**info.hf_overrides, **MOE_ON_EVERY_STAGE.get(arch, {})}
    with vllm_runner(
        info.default,
        revision=info.revision,
        trust_remote_code=info.trust_remote_code,
        dtype=info.dtype,
        load_format=LOAD_FORMAT,
        model_loader_extra_config={"perturb_layer": perturb_layer},
        hf_overrides=partial(
            dummy_hf_overrides,
            model_arch=arch,
            exist_overrides=overrides,
            num_hidden_layers=num_layers,
        ),
        pipeline_parallel_size=pp_size,
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=4,
        # The model's own default, which sparse MLA backends need.
        block_size=None,
        # Autotuned kernel choices can differ between engines; the defaults do not.
        enable_flashinfer_autotune=False,
        enable_prefix_caching=False,
        worker_extension_cls=f"{__name__}.LoaderInWorkers",
    ) as vllm_model:
        llm = vllm_model.llm
        vocab_size = min(llm.llm_engine.model_config.get_vocab_size(), 32000)
        rng = np.random.default_rng(0)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            ignore_eos=True,
            prompt_logprobs=0,
            logprobs=0,
            detokenize=False,
        )
        results = []
        # One request at a time, so that every layout runs the same shapes.
        for prompt_len in PROMPT_LENS:
            token_ids = rng.integers(5, vocab_size - 1, size=prompt_len).tolist()
            (output,) = llm.generate(
                TokensPrompt(prompt_token_ids=token_ids), params, use_tqdm=False
            )
            results.append(
                {
                    "prompt_logprobs": [
                        lp[tok].logprob
                        for tok, lp in zip(token_ids[1:], output.prompt_logprobs[1:])
                    ],
                    "tokens": list(output.outputs[0].token_ids),
                }
            )
        return results


def _max_logprob_diff(ref: list[dict], test: list[dict]) -> float:
    return max(
        abs(a - b)
        for r, t in zip(ref, test)
        for a, b in zip(r["prompt_logprobs"], t["prompt_logprobs"])
    )


def _check_equivalent(vllm_runner, arch: str, pp_size: int) -> None:
    num_layers = NUM_LAYERS[pp_size]
    ref = _run(vllm_runner, arch, 1, num_layers)
    test = _run(vllm_runner, arch, pp_size, num_layers)
    assert all(math.isfinite(x) for r in ref for x in r["prompt_logprobs"])
    # Same weights and batch shapes: every correct layout is bitwise equal, and
    # a single changed weight can move the logprobs by as little as 0.1.
    assert ref == test, (
        f"PP={pp_size} prompt logprobs differ by up to {_max_logprob_diff(ref, test)}"
    )


def _params(archs):
    params = skip_unavailable_models(archs, HF_EXAMPLE_MODELS.get_hf_info)
    return [
        pytest.param(
            p, marks=pytest.mark.xfail(strict=True, reason=KNOWN_PP_DEFECTS[p])
        )
        if isinstance(p, str) and p in KNOWN_PP_DEFECTS
        else p
        for p in params
    ]


@pytest.mark.timeout(CASE_TIMEOUT_S)
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("arch", _params(MINIMAL_PP_ARCH_LIST))
def test_pp_equivalence_small_subset(vllm_runner, arch: str):
    _check_equivalent(vllm_runner, arch, 2)


@pytest.mark.timeout(CASE_TIMEOUT_S)
@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("arch", _params(PP_ARCHS))
def test_pp_equivalence_middle_stages(vllm_runner, arch: str):
    """PP=4: two middle stages, which both receive and send."""
    _check_equivalent(vllm_runner, arch, 4)


@multi_gpu_test(num_gpus=2)
def test_pp_equivalence_detects_a_changed_weight(vllm_runner):
    """Negative control: a weight that differs on the last stage must fail."""
    arch, num_layers = MINIMAL_PP_ARCH_LIST[0], NUM_LAYERS[2]
    ref = _run(vllm_runner, arch, 1, num_layers)
    test = _run(vllm_runner, arch, 2, num_layers, perturb_layer=num_layers - 1)
    assert ref != test


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("arch", SPLIT_KV_SHARING)
def test_pp_rejects_split_kv_sharing(vllm_runner, monkeypatch, arch: str):
    """A split that separates a layer from the layer whose KV cache it reads must
    fail at startup rather than produce output."""
    num_layers, partition = SPLIT_KV_SHARING[arch]
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    with pytest.raises(RuntimeError, match="Engine core initialization failed"):
        _run(vllm_runner, arch, 2, num_layers)
