# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the functionality of the Transformers modeling backend."""

import contextlib
import os
import tempfile
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.multimodal import MultiModalMixin
from vllm.model_executor.models.utils import StageMissingLayer

from ...conftest import HfRunner, VllmRunner
from ...utils import multi_gpu_test, prep_prompts
from ..registry import HF_EXAMPLE_MODELS
from ..utils import check_embeddings_close, check_logprobs_close


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def get_model(arch: str) -> str:
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_transformers_version(on_fail="skip")
    return model_info.default


def get_num_fused(model) -> tuple[int, int]:
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    glu = sum(isinstance(m, MergedColumnParallelLinear) for m in model.modules())
    qkv = sum(isinstance(m, QKVParallelLinear) for m in model.modules())
    return glu, qkv


def count_mla_layers(model) -> int:
    from vllm.model_executor.layers.attention import MLAAttention

    return sum(isinstance(m, MLAAttention) for m in model.attention_instances.values())


def check_implementation(
    runner_ref: type[HfRunner | VllmRunner],
    runner_test: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    kwargs_ref: dict[str, Any] | None = None,
    kwargs_test: dict[str, Any] | None = None,
    num_fused: tuple[int, int] = (1, 1),
    **kwargs,
):
    if kwargs_ref is None:
        kwargs_ref = {}
    if kwargs_test is None:
        kwargs_test = {}

    max_tokens = 32
    num_logprobs = 5

    args = (example_prompts, max_tokens, num_logprobs)

    with runner_test(model, **kwargs_test, **kwargs) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        num_layers = model_config.hf_config.get_text_config().num_hidden_layers
        expected_glu, expected_qkv = num_fused
        for num_glu, num_qkv in model_test.apply_model(get_num_fused):
            assert num_glu == expected_glu * num_layers
            assert num_qkv == expected_qkv * num_layers

        outputs_test = model_test.generate_greedy_logprobs(*args)

    with runner_ref(model, **kwargs_ref) as model_ref:
        if isinstance(model_ref, VllmRunner):
            outputs_ref = model_ref.generate_greedy_logprobs(*args)
        else:
            outputs_ref = model_ref.generate_greedy_logprobs_limit(*args)

    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="ref",
        name_1="test",
    )


@pytest.mark.parametrize(
    "model,model_impl,num_fused",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", "transformers", (1, 1)),
        ("hmellor/Ilama-3.2-1B", "auto", (1, 1)),  # CUSTOM CODE
        ("allenai/OLMoE-1B-7B-0924", "transformers", (0, 1)),  # MoE
    ],
)  # trust_remote_code=True by default
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    model_impl: str,
    num_fused: tuple[int, int],
) -> None:
    check_implementation(
        hf_runner,
        vllm_runner,
        example_prompts,
        model,
        num_fused=num_fused,
        model_impl=model_impl,
    )


def test_hybrid_attention(vllm_runner: type[VllmRunner]) -> None:
    prompts, _, _ = prep_prompts(4, (800, 801))
    kwargs_ref = {"max_model_len": 8192, "enforce_eager": True}
    kwargs_test = {"model_impl": "transformers", **kwargs_ref}
    check_implementation(
        vllm_runner,
        vllm_runner,
        prompts,
        model="hmellor/tiny-random-Gemma2ForCausalLM",
        kwargs_ref=kwargs_ref,
        kwargs_test=kwargs_test,
    )


def test_mla(vllm_runner: type[VllmRunner], example_prompts: list[str]) -> None:
    import transformers
    from packaging.version import Version

    installed = Version(transformers.__version__)
    required = Version("5.15.0.dev0")
    if installed < required:
        pytest.skip(
            "MLA models with the Transformers modeling backend require "
            f"transformers>={required}, but got {installed}"
        )

    model = "hmellor/tiny-random-DeepseekV2ForCausalLM"
    args = (example_prompts, 32, 5)
    kwargs: dict[str, Any] = {"max_model_len": 2048, "enforce_eager": True}

    with vllm_runner(
        model, model_impl="transformers", trust_remote_code=False, **kwargs
    ) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()
        num_layers = model_config.hf_config.get_text_config().num_hidden_layers
        assert model_test.apply_model(count_mla_layers) == [num_layers]
        outputs_test = model_test.generate_greedy_logprobs(*args)

    with vllm_runner(model, model_impl="auto") as model_ref:
        outputs_ref = model_ref.generate_greedy_logprobs(*args)

    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="native",
        name_1="transformers",
    )


@multi_gpu_test(num_gpus=2)
def test_distributed(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts,
):
    kwargs = {"model_impl": "transformers", "tensor_parallel_size": 2}
    check_implementation(
        hf_runner,
        vllm_runner,
        example_prompts,
        "meta-llama/Llama-3.2-1B-Instruct",
        kwargs_test=kwargs,
    )


@pytest.mark.parametrize(
    "model, quantization_kwargs",
    [
        ("TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ", {}),
        ("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ", {}),
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_quantization(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    quantization_kwargs: dict[str, str],
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        model,
        model_impl="auto",
        enforce_eager=True,
        **quantization_kwargs,  # type: ignore[arg-type]
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    with vllm_runner(
        model,
        model_impl="transformers",
        enforce_eager=True,
        **quantization_kwargs,  # type: ignore[arg-type]
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        transformers_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=transformers_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="transformers",
        name_1="vllm",
    )


@pytest.mark.parametrize(
    "model",
    [
        # Layers live in `layers`
        "Qwen/Qwen3-Embedding-0.6B",
        # Layers live in `model.layers`
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_embed_loading(vllm_runner, model):
    with vllm_runner(
        model,
        max_model_len=1024,
        enforce_eager=True,
        runner="pooling",
        model_impl="transformers",
    ) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()


@pytest.mark.parametrize(
    "arch", ["TransformersEmbeddingModel", "TransformersForSequenceClassification"]
)
def test_pooling(hf_runner, vllm_runner, example_prompts, arch):
    model = get_model(arch)

    vllm_kwargs = dict(max_model_len=None, model_impl="transformers")

    hf_kwargs = dict()
    if arch == "TransformersEmbeddingModel":
        hf_kwargs["is_sentence_transformer"] = True
    elif arch == "TransformersForSequenceClassification":
        from transformers import AutoModelForSequenceClassification

        hf_kwargs["auto_cls"] = AutoModelForSequenceClassification

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    with (
        vllm_runner(model, **vllm_kwargs) as vllm_model,
        hf_runner(model, **hf_kwargs) as hf_model,
    ):
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        if arch == "TransformersEmbeddingModel":
            vllm_outputs = vllm_model.embed(example_prompts)
            hf_outputs = hf_model.encode(example_prompts)
        elif arch == "TransformersForSequenceClassification":
            vllm_outputs = vllm_model.classify(example_prompts)
            hf_outputs = hf_model.classify(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


VOCAB_SIZE = 64
HIDDEN_SIZE = 8
EMBED_SCALE = 3.0


class ScaledWordEmbedding(nn.Embedding):
    """Mirrors Transformers' `*ScaledWordEmbedding` classes."""

    def __init__(
        self, num_embeddings, embedding_dim, padding_idx=None, embed_scale=1.0
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class ComposedWordEmbedding(nn.Module):
    """Scales embeddings, but wraps `nn.Embedding` instead of inheriting from it."""

    def __init__(self, num_embeddings, embedding_dim, embed_scale=1.0):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return self.embed(input_ids) * self.embed_scale


@pytest.fixture
def tp_init():
    """Single rank tensor parallel state, so vLLM layers can be constructed."""
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    from ...utils import ensure_current_vllm_config

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend=current_platform.dist_backend,
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


@pytest.fixture
def vpe(tp_init):
    """`VocabParallelEmbedding`, imported late so collection does not import vLLM."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    return VocabParallelEmbedding


def replace(embedding):
    """Replace `embedding` and fill the new weights with recognisable values."""
    from vllm.model_executor.models.transformers.utils import replace_embedding_class

    new_embedding = replace_embedding_class(embedding)
    for _, param in new_embedding.named_parameters():
        param.data = torch.arange(param.numel(), dtype=param.dtype).view(param.shape)
    return new_embedding


def assert_scaled(vpe, module, embedding=None):
    """`module`'s output is `embedding`'s unscaled output times `EMBED_SCALE`."""
    input_ids = torch.arange(VOCAB_SIZE)
    unscaled = vpe.forward(embedding if embedding is not None else module, input_ids)
    torch.testing.assert_close(module(input_ids), unscaled * EMBED_SCALE)


def test_replace_plain_embedding(vpe):
    """A plain `nn.Embedding` is replaced outright, leaving no subclass behind."""
    assert type(replace(nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE))) is vpe


def test_replace_infers_shape_and_dtype(tp_init):
    """Shape and dtype come from the replaced module, not from the config."""
    embedding = nn.Embedding(VOCAB_SIZE * 2, HIDDEN_SIZE + 1, dtype=torch.float16)
    new_embedding = replace(embedding)

    assert new_embedding.num_embeddings == VOCAB_SIZE * 2
    assert new_embedding.org_vocab_size == VOCAB_SIZE * 2
    assert new_embedding.embedding_dim == HIDDEN_SIZE + 1
    assert new_embedding.weight.dtype == torch.float16


def test_replace_inherited_embedding(vpe):
    """Subclasses keep their extra state and their scaled `forward`."""
    new_embedding = replace(
        ScaledWordEmbedding(
            VOCAB_SIZE, HIDDEN_SIZE, padding_idx=0, embed_scale=EMBED_SCALE
        )
    )

    assert isinstance(new_embedding, vpe)
    assert new_embedding.scalar_embed_scale == EMBED_SCALE
    assert "embed_scale" in new_embedding._non_persistent_buffers_set
    assert_scaled(vpe, new_embedding)


def test_replace_composed_embedding(vpe):
    """Wrappers are left alone; only the `nn.Embedding` they hold is replaced."""
    embedding = ComposedWordEmbedding(VOCAB_SIZE, HIDDEN_SIZE, embed_scale=EMBED_SCALE)
    new_embedding = replace(embedding)

    assert new_embedding is embedding
    assert type(embedding.embed) is vpe
    assert_scaled(vpe, new_embedding, embedding.embed)


def test_replace_nested_embedding(vpe):
    """The composed `nn.Embedding` is found and set however deeply it is nested."""
    wrapper = nn.Module()
    wrapper.add_module("inner", nn.Module())
    wrapper.inner.add_module(
        "embed", ScaledWordEmbedding(VOCAB_SIZE, HIDDEN_SIZE, embed_scale=EMBED_SCALE)
    )
    replace(wrapper)

    assert isinstance(wrapper.inner.embed, vpe)
    assert_scaled(vpe, wrapper.inner.embed)


@pytest.mark.parametrize("num_embeddings", [0, 2])
def test_replace_ambiguous_embedding(tp_init, num_embeddings):
    """Composing anything but one `nn.Embedding` is an error, not a silent guess."""
    wrapper = nn.Module()
    for i in range(num_embeddings):
        wrapper.add_module(f"embed_{i}", nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE))

    with pytest.raises(ValueError, match=f"found {num_embeddings}"):
        replace(wrapper)


def test_replaced_embedding_exposes_one_vpe(vpe):
    """`CausalMixin` ties `lm_head` to the one `VocabParallelEmbedding` it can find.

    `tie_weights` reads `.weight`, which a composing module does not have, so it must
    be handed the composed embedding instead.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    inherited = replace(ScaledWordEmbedding(VOCAB_SIZE, HIDDEN_SIZE))
    composed = replace(ComposedWordEmbedding(VOCAB_SIZE, HIDDEN_SIZE, EMBED_SCALE))

    assert [m for m in inherited.modules() if isinstance(m, vpe)] == [inherited]
    assert [m for m in composed.modules() if isinstance(m, vpe)] == [composed.embed]

    lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE)
    assert lm_head.tie_weights(composed.embed).weight is composed.embed.weight


MULTIMODAL_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


class MarkingStub(SupportsMultiModal, nn.Module):
    """Just enough of the backend to exercise component marking."""

    _mark_model_components = MultiModalMixin._mark_model_components
    _find_encoder_classes = MultiModalMixin._find_encoder_classes
    _from_config_kwargs = Base._from_config_kwargs
    _pre_trained_model_classes = Base._pre_trained_model_classes


def build_marked_model(image_limit: int, skip_tokenizer_init: bool = False):
    """Build the HF model inside the marking context and return it."""
    model_config = ModelConfig(
        model=MULTIMODAL_MODEL,
        model_impl="transformers",
        limit_mm_per_prompt={"image": image_limit},
    )
    # Set after construction: building the config itself needs the tokenizer
    model_config.skip_tokenizer_init = skip_tokenizer_init
    stub = MarkingStub()
    stub.config = AutoConfig.from_pretrained(MULTIMODAL_MODEL)
    stub.model_config = model_config

    vllm_config = VllmConfig(model_config=model_config)
    with stub._mark_model_components(vllm_config), torch.device("meta"):
        stub.model = AutoModel.from_config(**stub._from_config_kwargs)
    return stub.model


@pytest.mark.parametrize(("image_limit", "skipped"), [(0, True), (4, False)])
def test_tower_weights_skipped_when_modality_disabled(image_limit, skipped):
    """`--limit-mm-per-prompt image=0` should drop the vision tower's weights."""
    vision_tower = build_marked_model(image_limit).vision_tower
    assert isinstance(vision_tower, StageMissingLayer) is skipped


def test_marking_skipped_without_tokenizer():
    """Marking needs the HF processor, which needs a tokenizer, so it is skipped.

    The tower is built as normal rather than the model failing to load.
    """
    vision_tower = build_marked_model(0, skip_tokenizer_init=True).vision_tower
    assert not isinstance(vision_tower, StageMissingLayer)
