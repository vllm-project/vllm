# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the functionality of the Transformers modeling backend."""

import ast
import contextlib
import os
import tempfile
from collections import Counter
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    PreTrainedConfig,
    core_model_loading,
)
from transformers.conversion_mapping import WeightConverter, WeightRenaming

from vllm.compilation.counter import compilation_counter
from vllm.config import (
    ModelConfig,
    ParallelConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode
from vllm.model_executor import model_loader
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.fusers import AttentionFuser
from vllm.model_executor.models.transformers.fusers.attention import VLLM_ATTN_IMPL
from vllm.model_executor.models.transformers.multimodal import MultiModalMixin
from vllm.model_executor.models.transformers.utils import infer_dynamic_arg_dims
from vllm.model_executor.models.utils import (
    ShardId,
    StageMissingLayer,
    WeightsMapper,
    _model_to_pp_missing_layer_names,
)

from ...conftest import HfRunner, VllmRunner
from ...utils import create_new_process_for_each_test, multi_gpu_test, prep_prompts
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


def count_modules(model, cls: type) -> int:
    return sum(isinstance(m, cls) for m in model.modules())


def count_fusers(model) -> dict[str, int]:
    counts = Counter(type(f).__name__ for fs in model.fusers.values() for f in fs)
    return dict(counts)


def check_implementation(
    runner_ref: type[HfRunner | VllmRunner],
    runner_test: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    kwargs_ref: dict[str, Any] | None = None,
    kwargs_test: dict[str, Any] | None = None,
    num_fused: dict[str, int] | None = None,
    **kwargs,
):
    if num_fused is None:
        num_fused = {"GLUFuser": 1, "QKVFuser": 1}
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
        expected = {name: n * num_layers for name, n in num_fused.items()}
        for counts in model_test.apply_model(count_fusers):
            assert {name: counts.get(name, 0) for name in expected} == expected

        tp_size = kwargs_test.get("tensor_parallel_size", 1)
        from vllm.model_executor.layers.attention import Attention

        counts = model_test.apply_model(partial(count_modules, cls=Attention))
        assert counts == [num_layers] * tp_size

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
    "model,model_impl,num_fused,trust_remote_code",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", "transformers", None, False),
        ("vllm-project/Ilama-3.2-1B", "auto", None, True),
        ("allenai/OLMoE-1B-7B-0924", "transformers", {"QKVFuser": 1}, False),  # MoE
    ],
)
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    model_impl: str,
    num_fused: dict[str, int] | None,
    trust_remote_code: bool,
) -> None:
    check_implementation(
        hf_runner,
        vllm_runner,
        example_prompts,
        model,
        kwargs_ref={"trust_remote_code": trust_remote_code},
        num_fused=num_fused,
        model_impl=model_impl,
        trust_remote_code=trust_remote_code,
    )


@create_new_process_for_each_test("spawn")
def test_compile_mm_encoder(
    vllm_runner: type[VllmRunner], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idefics3's encoder has an unannotated and a `torch.BoolTensor` input."""
    # Disable multiprocessing so that the counter is in the same process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # One graph for the language model and one for the encoder
    with (
        compilation_counter.expect(num_models_seen=2),
        vllm_runner(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            model_impl="transformers",
            max_model_len=2048,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.5,
            compilation_config={
                "mode": CompilationMode.VLLM_COMPILE,
                "compile_mm_encoder": True,
            },
        ),
    ):
        pass


def test_eagle3_mm(vllm_runner: type[VllmRunner], tmp_path) -> None:
    """EAGLE-3 runs on a composite VLM whose decoder records the aux hidden states."""
    model = "HuggingFaceTB/SmolVLM-256M-Instruct"
    draft_config = AutoConfig.from_pretrained(model).text_config
    draft_config.architectures = ["LlamaForCausalLMEagle3"]
    draft_config.num_hidden_layers = 1
    draft_config.save_pretrained(tmp_path)

    with vllm_runner(
        model,
        model_impl="transformers",
        load_format="dummy",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.5,
        speculative_config={
            "method": "eagle3",
            "model": str(tmp_path),
            "num_speculative_tokens": 2,
        },
    ) as llm:
        llm.generate_greedy(["Hello"], max_tokens=8)


def test_infer_dynamic_arg_dims():
    """Unannotated and tensor arguments are dynamic, unless a non-`None` default
    means they are passed as a plain value (e.g. Gemma 4's `logits_to_keep=0`)."""

    class Encoder(nn.Module):
        def forward(
            self,
            pixel_values,
            position_ids: torch.Tensor,
            mask: torch.BoolTensor | None = None,
            grid: tuple[int, int] = (1, 1),
            tiles: list[torch.Tensor] | None = None,
            use_cache=False,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs,
        ):
            pass

    dynamic = ("pixel_values", "position_ids", "mask")
    assert infer_dynamic_arg_dims(Encoder) == dict.fromkeys(dynamic, 0)
    assert infer_dynamic_arg_dims(Encoder, dim=1) == dict.fromkeys(dynamic, 1)


def test_hybrid_attention(vllm_runner: type[VllmRunner]) -> None:
    prompts, _, _ = prep_prompts(4, (800, 801))
    kwargs_ref = {"max_model_len": 8192, "enforce_eager": True}
    kwargs_test = {"model_impl": "transformers", **kwargs_ref}
    check_implementation(
        vllm_runner,
        vllm_runner,
        prompts,
        model="vllm-project/tiny-random-Gemma2ForCausalLM",
        kwargs_ref=kwargs_ref,
        kwargs_test=kwargs_test,
    )


def test_whole_graph_compile(
    vllm_runner: type[VllmRunner], example_prompts: list[str]
) -> None:
    """Compiling attention and the KV-cache op into one Inductor graph must
    match eager, so vllm:: ops have to receive eager strides."""
    kwargs_ref = {"model_impl": "transformers", "max_model_len": 2048}
    kwargs_test = {
        **kwargs_ref,
        "compilation_config": {"mode": 2, "backend": "inductor"},
    }
    check_implementation(
        vllm_runner,
        vllm_runner,
        example_prompts,
        model="Qwen/Qwen3-0.6B",
        kwargs_ref={**kwargs_ref, "enforce_eager": True},
        kwargs_test=kwargs_test,
    )


def get_sinks(model) -> dict[int, torch.Tensor]:
    """The sink tensor each attention layer was handed, keyed by layer index."""
    sinks = {}
    for i, (prefix, fuser) in model.attention_fusers.items():
        if (sink := fuser.sinks(model.get_submodule(prefix))) is not None:
            sinks[i] = sink.float().cpu()
    return sinks


def test_sinks(hf_runner: type[HfRunner], vllm_runner: type[VllmRunner]) -> None:
    """Learnable attention sinks must reach the attention layers.

    Only the attention impl can apply them, so if they are not passed to
    `Attention` they are dropped and every softmax is subtly wrong.
    """
    from vllm.platforms import current_platform

    if not (current_platform.is_xpu() or current_platform.has_device_capability(90)):
        pytest.skip("Attention sinks need FlashAttention 3, TRTLLM or XPU attention")

    model = "tiny-random/gpt-oss-bf16"
    with vllm_runner(
        model, model_impl="transformers", max_model_len=1024, enforce_eager=True
    ) as model_test:
        assert model_test.llm.llm_engine.model_config.using_transformers_backend()
        sinks = model_test.apply_model(get_sinks)[0]

    with hf_runner(model, dtype="bfloat16") as model_ref:
        layers = model_ref.model.model.get_decoder().layers
        expected = {i: layer.self_attn.sinks.float() for i, layer in enumerate(layers)}

    assert sinks.keys() == expected.keys()
    for i, sink in sinks.items():
        torch.testing.assert_close(sink, expected[i].cpu())


def test_mla(vllm_runner: type[VllmRunner], example_prompts: list[str]) -> None:
    model = "vllm-project/tiny-random-DeepseekV2ForCausalLM"
    args = (example_prompts, 32, 5)
    kwargs: dict[str, Any] = {"max_model_len": 2048, "enforce_eager": True}

    with vllm_runner(model, model_impl="transformers", **kwargs) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()
        num_layers = model_config.hf_config.get_text_config().num_hidden_layers
        from vllm.model_executor.layers.attention import MLAAttention

        counts = model_test.apply_model(partial(count_modules, cls=MLAAttention))
        assert counts == [num_layers]
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
@pytest.mark.parametrize(
    "model, num_fused",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", None),
        # Packed qkv_proj sliced with global offsets in the forward
        ("microsoft/Phi-4-mini-instruct", {"PackedQKVFuser": 1}),
    ],
)
def test_distributed(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts,
    model: str,
    num_fused: dict[str, int] | None,
):
    kwargs = {"model_impl": "transformers", "tensor_parallel_size": 2}
    check_implementation(
        hf_runner,
        vllm_runner,
        example_prompts,
        model,
        kwargs_test=kwargs,
        num_fused=num_fused,
    )


@pytest.mark.parametrize(
    "model, quantization_kwargs",
    [
        ("vllm-project/TinyLlama-1.1B-Chat-v0.3-AWQ", {}),
        ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", {}),
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
        **quantization_kwargs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    with vllm_runner(
        model,
        model_impl="transformers",
        enforce_eager=True,
        **quantization_kwargs,
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
def test_pooling(
    hf_runner,
    vllm_runner,
    example_prompts,
    arch,
):
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
PER_LAYER_VOCAB_SIZE = 32
NUM_POSITIONS = 16
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


def assert_scaled(vpe, module):
    """`module`'s output is its own unscaled output times `EMBED_SCALE`."""
    input_ids = torch.arange(VOCAB_SIZE)
    unscaled = vpe.forward(module, input_ids)
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


def test_replace_is_idempotent(tp_init):
    """A tied table is reached twice by the recursion, and must not be rebased twice."""
    from vllm.model_executor.models.transformers.utils import replace_embedding_class

    new_embedding = replace(ScaledWordEmbedding(VOCAB_SIZE, HIDDEN_SIZE, embed_scale=2))
    again = replace_embedding_class(new_embedding)

    assert again is new_embedding
    assert type(again) is type(new_embedding)


class FakeVocabModel(nn.Module):
    """The embedding tables a model can hold, only one of which is a vocab table.

    `embed_tokens_per_layer` mirrors Gemma 3n, whose per-layer table is sized by
    `vocab_size_per_layer_input` rather than `vocab_size`.
    """

    def __init__(self, per_layer: bool = False, composed: bool = False):
        super().__init__()
        embed_cls = ComposedWordEmbedding if composed else ScaledWordEmbedding
        self.embed_tokens = embed_cls(VOCAB_SIZE, HIDDEN_SIZE, embed_scale=EMBED_SCALE)
        if per_layer:
            self.embed_tokens_per_layer = ScaledWordEmbedding(
                PER_LAYER_VOCAB_SIZE, HIDDEN_SIZE, embed_scale=EMBED_SCALE
            )
        # Not a vocab table: indexed by position, and read as a whole by models
        # which interpolate it (e.g. `SiglipVisionEmbeddings`)
        self.position_embedding = nn.Embedding(NUM_POSITIONS, HIDDEN_SIZE)

    def get_input_embeddings(self):
        return self.embed_tokens


def replace_vocab_embeddings(model, **config_kwargs):
    """Replace `model`'s vocab tables as `recursive_replace` would, and return them."""
    from vllm.model_executor.models.transformers.utils import attrsetter

    stub = nn.Module()
    stub.model = model
    stub.config = PreTrainedConfig(
        vocab_size=VOCAB_SIZE, num_positions=NUM_POSITIONS, **config_kwargs
    )
    embeddings = Base._vocab_embeddings(stub)

    replaced = []
    for name, module in list(model.named_modules()):
        if module in embeddings:
            replaced.append(replace(module))
            attrsetter(name)(model, replaced[-1])
    return replaced


def test_only_vocab_tables_are_replaced(vpe):
    """Position embeddings share the `nn.Embedding` type, but not the treatment."""
    model = FakeVocabModel()

    assert replace_vocab_embeddings(model) == [model.embed_tokens]
    assert isinstance(model.embed_tokens, vpe)
    assert type(model.position_embedding) is nn.Embedding


def test_per_layer_vocab_table_is_replaced(vpe):
    """A second vocab table is found by its size, without a bespoke accessor."""
    model = FakeVocabModel(per_layer=True)
    replaced = replace_vocab_embeddings(
        model, vocab_size_per_layer_input=PER_LAYER_VOCAB_SIZE
    )

    assert replaced == [model.embed_tokens, model.embed_tokens_per_layer]
    assert all(isinstance(module, vpe) for module in replaced)


def test_composed_input_embeddings_are_replaced(vpe):
    """A wrapper is left alone; only the `nn.Embedding` it composes is replaced.

    `CausalMixin` ties `lm_head` to whichever `VocabParallelEmbedding` it finds under
    `get_input_embeddings()`, reading a `.weight` the wrapper does not have.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    model = FakeVocabModel(composed=True)
    wrapper = model.embed_tokens
    replaced = replace_vocab_embeddings(model)

    assert model.embed_tokens is wrapper
    assert replaced == [m for m in wrapper.modules() if isinstance(m, vpe)]

    lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE)
    assert lm_head.tie_weights(replaced[0]).weight is replaced[0].weight


def test_missing_input_embeddings_are_skipped(tp_init):
    """Pipeline ranks without the embeddings have a `PPMissingLayer` in their place."""
    from vllm.model_executor.models.utils import PPMissingLayer

    model = FakeVocabModel()
    model.embed_tokens = PPMissingLayer()

    assert replace_vocab_embeddings(model) == []


MULTIMODAL_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


class MarkingStub(SupportsMultiModal, nn.Module):
    """Just enough of the backend to exercise component marking."""

    _mark_model_components = MultiModalMixin._mark_model_components
    _find_encoder_classes = MultiModalMixin._find_encoder_classes
    _from_config_kwargs = Base._from_config_kwargs
    _pre_trained_model_classes = Base._pre_trained_model_classes


def build_marked_model(
    image_limit: int, video_limit: int = 0, skip_tokenizer_init: bool = False
):
    """Build the HF model inside the marking context and return it."""
    model_config = ModelConfig(
        model=MULTIMODAL_MODEL,
        model_impl="transformers",
        limit_mm_per_prompt={"image": image_limit, "video": video_limit},
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


@pytest.mark.parametrize(
    ("image_limit", "video_limit", "skipped"),
    [(0, 0, True), (4, 0, False), (0, 4, False)],
)
def test_tower_weights_skipped_when_modality_disabled(
    image_limit, video_limit, skipped, tp_init
):
    """Only `--limit-mm-per-prompt image=0,video=0` drops the tower they share."""
    vision_tower = build_marked_model(image_limit, video_limit).vision_tower
    assert isinstance(vision_tower, StageMissingLayer) is skipped


def test_marking_skipped_without_tokenizer():
    """Marking needs the HF processor, which needs a tokenizer, so it is skipped.

    The tower is built as normal rather than the model failing to load.
    """
    vision_tower = build_marked_model(0, skip_tokenizer_init=True).vision_tower
    assert not isinstance(vision_tower, StageMissingLayer)


def test_missing_encoders_are_skipped():
    """Gemma 4 31B has no audio tower, so `get_encoder` returns `None` for audio."""

    class Model(nn.Module):
        def get_encoder(self, modality: str) -> nn.Module | None:
            return nn.Identity() if modality == "image" else None

    assert MarkingStub()._find_encoder_classes(Model()) == {"image": nn.Identity}


NUM_LAYERS = 4


def build_model(model_type: str, **overrides) -> nn.Module:
    """A tiny HF model of `NUM_LAYERS` layers, built on the meta device."""
    try:
        config = AutoConfig.for_model(
            model_type,
            num_hidden_layers=NUM_LAYERS,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=VOCAB_SIZE,
            **overrides,
        )
    except ValueError:
        pytest.skip(f"The installed transformers has no {model_type!r} model")
    with torch.device("meta"):
        return AutoModel.from_config(config)


ATTENTION_MODEL_TYPES = [
    # The default `head_size**-0.5` happens to be right here
    "llama",
    # Scales the query by a learnable per-dim weight, so it declares 1.0
    "gemma4_text",
    # MLA, and declares yarn's `mscale`
    "deepseek_v3",
]


@pytest.mark.parametrize("model_type", ATTENTION_MODEL_TYPES)
def test_attention_dispatch_is_matched(model_type: str):
    """Exactly the decoder layers' attention modules match an `AttentionFuser`.

    That match is what `recursive_replace` records, and so what
    `_create_attention_instances` attaches vLLM's attention layer to.
    """
    model = build_model(model_type)
    matched = {
        name
        for name, module in model.named_modules()
        if AttentionFuser.match(None, module) is not None
    }
    assert matched == {f"layers.{i}.self_attn" for i in range(NUM_LAYERS)}


def test_attention_layer_index_is_the_modules_own():
    """The layer served comes from the module, not from its position in the stack.

    LongCat Flash gives each decoder layer two attention sublayers numbered
    `2i` and `2i + 1`, so `num_hidden_layers` is twice the length of the stack
    and the enclosing layer's position is not the index the KV cache is keyed by.
    A vision tower is excluded separately, by `validate`: only the text config is
    patched to dispatch to vLLM, so a tower's attention is left to Transformers.
    """
    model = build_model("longcat_flash", num_layers=2)
    text_config = model.config.get_text_config()
    assert len(model.layers) * 2 == text_config.num_hidden_layers

    attentions = [
        model.get_submodule(f"layers.{i}.self_attn.{j}") for i in (0, 1) for j in (0, 1)
    ]
    fusers: list[AttentionFuser] = []
    for attn in attentions:
        fuser = AttentionFuser.match(None, attn)
        assert fuser is not None
        fusers.append(fuser)
    assert [f.layer_index(a) for f, a in zip(fusers, attentions)] == [0, 1, 2, 3]

    # Not dispatching to vLLM yet, as a vision tower never would be. The vLLM
    # config is not consulted, only the one the module was built with.
    assert not any(f.validate(a, None) for f, a in zip(fusers, attentions))
    text_config._attn_implementation = VLLM_ATTN_IMPL
    assert all(f.validate(a, None) for f, a in zip(fusers, attentions))


def test_attention_dispatch_is_required():
    """Models selected by the permissive registry fail clearly without dispatch."""
    model = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=1),
        attention_fusers={},
        _get_attn_cls=lambda: None,
        _pp_indices=lambda: (0, 1),
    )

    with pytest.raises(
        ValueError,
        match="Layer 0 does not dispatch through the Transformers attention interface",
    ):
        Base._create_attention_instances(model)


@pytest.mark.parametrize("model_type", ATTENTION_MODEL_TYPES)
def test_attention_scale_is_the_declared_one(model_type: str):
    """The scale comes off the module, not from the `head_size**-0.5` default.

    Two of the three model types declare something the default would silently
    replace, so they are what makes this more than a tautology.
    """
    attention = build_model(model_type).layers[0].self_attn
    fuser = AttentionFuser.match(None, attention)
    assert fuser is not None
    assert fuser.scale(attention) == attention.scaling


def test_attention_scale_is_the_argument_not_the_attribute():
    """OPT applies the scale to the query itself, then declares a literal 1.0.

    Its `self.scaling` is the `head_size**-0.5` it has already applied, so
    reading the attribute rather than the argument the module hands the
    interface would scale twice.
    """
    model = build_model("opt", ffn_dim=128, word_embed_proj_dim=64)
    attention = model.get_submodule("decoder.layers.0.self_attn")
    fuser = AttentionFuser.match(None, attention)
    assert fuser is not None
    assert fuser.scale(attention) == 1.0 != attention.scaling


def test_attention_scale_rejects_unresolvable_expression():
    """Unresolvable declared scales must not silently use the Llama default."""

    class Attention(nn.Module):
        head_dim = 16

    fuser = AttentionFuser(
        source_cls="Attention",
        scale_expr=ast.parse("self.head_dim**-0.5 * self.factor", mode="eval").body,
    )

    with pytest.raises(ValueError, match="Cannot resolve attention scaling expression"):
        fuser.scale(Attention())


class _ImageOnlyMRoPEModel:
    """`get_rope_index` without `video_grid_thw` or `**kwargs` (e.g. HunYuanVL)."""

    def get_rope_index(self, input_ids, image_grid_thw):
        seq_len = input_ids.shape[-1]
        positions = torch.arange(seq_len).view(1, 1, -1).expand(4, 1, -1)
        return positions, torch.tensor([0])


def test_get_mrope_input_positions_omits_unsupported_grid_kwargs():
    """Optional grids the model can't accept must not be passed when empty."""
    mixin = SimpleNamespace(model=_ImageOnlyMRoPEModel())

    positions, delta = MultiModalMixin.get_mrope_input_positions(mixin, [1, 2, 3], [])

    assert positions.shape == (4, 3)
    assert delta == 0


# The smallest sizes the fused MoE kernels accept
MTP_HIDDEN_SIZE = 128
MTP_MOE_KWARGS = dict(
    vocab_size=128,
    hidden_size=MTP_HIDDEN_SIZE,
    intermediate_size=256,
    moe_intermediate_size=128,
    n_routed_experts=4,
    num_experts_per_tok=2,
)
# Real layer counts, since the MTP weight patterns hard-code them
MTP_MODELS = {
    "glm4_moe": dict(
        num_hidden_layers=46, num_attention_heads=2, num_key_value_heads=1, head_dim=64
    ),
    "deepseek_v3": dict(
        num_hidden_layers=61,
        num_attention_heads=16,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
    ),
}


@pytest.mark.parametrize("model_type", MTP_MODELS)
def test_mtp_draft_loads_mtp_weights(model_type, tmp_path, tp_init, monkeypatch):
    """An MTP draft for a model following the Transformers MTP standard loads each
    MTP layer appended to the target's checkpoint into `MtpModel`'s layout."""
    kwargs = MTP_MODELS[model_type] | MTP_MOE_KWARGS
    num_hidden_layers = kwargs.pop("num_hidden_layers")
    num_mtp_layers = 2
    hidden_size = MTP_HIDDEN_SIZE
    config = AutoConfig.for_model(
        model_type, num_hidden_layers=num_hidden_layers + num_mtp_layers, **kwargs
    )
    model_cls = AutoModelForCausalLM._model_mapping[type(config)]
    config.architectures = [model_cls.__name__]
    # These have one MTP layer, so register the second like Step-3.7 registers its own
    monkeypatch.setattr(
        model_cls,
        "_keys_to_ignore_on_load_unexpected",
        [rf"model\.layers\.{num_hidden_layers + 1}.*"]
        + model_cls._keys_to_ignore_on_load_unexpected,
    )
    # Save the MTP layers as the last decoder layers, then add their MTP modules
    model_cls(config).save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    for i in range(num_hidden_layers, num_hidden_layers + num_mtp_layers):
        for name, shape in {
            "enorm.weight": (hidden_size,),
            "hnorm.weight": (hidden_size,),
            "eh_proj.weight": (hidden_size, 2 * hidden_size),
            "shared_head.norm.weight": (hidden_size,),
        }.items():
            weights[f"model.layers.{i}.{name}"] = torch.randn(shape)
    save_file(weights, tmp_path / "model.safetensors", metadata={"format": "pt"})
    config.num_hidden_layers = num_hidden_layers
    config.num_nextn_predict_layers = num_mtp_layers
    config.save_pretrained(tmp_path)

    target = ModelConfig(str(tmp_path), model_impl="transformers", dtype="bfloat16")
    speculative_config = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=num_mtp_layers,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    draft = speculative_config.draft_model_config
    assert draft.architecture == "TransformersMoEForMTP"
    vllm_config = VllmConfig(model_config=target, speculative_config=speculative_config)
    with set_current_vllm_config(vllm_config):
        model = model_loader.get_model(vllm_config=vllm_config, model_config=draft)

    assert len(model.model.layers) == num_mtp_layers
    for k, layer in enumerate(model.model.layers):
        mtp = f"model.layers.{num_hidden_layers + k}"
        for ckpt_name, param_name in {
            "enorm.weight": "enorm.weight",
            "hnorm.weight": "hnorm.weight",
            "eh_proj.weight": "eh_proj.weight",
            "shared_head.norm.weight": "post_norm.weight",
            "input_layernorm.weight": "mtp_block.input_layernorm.weight",
            "self_attn.o_proj.weight": "mtp_block.self_attn.o_proj.weight",
        }.items():
            torch.testing.assert_close(
                layer.get_parameter(param_name).cpu(),
                weights[f"{mtp}.{ckpt_name}"].to(torch.bfloat16),
            )


def test_mtp_draft_for_multimodal_model(tmp_path, tp_init):
    """A multimodal model, which is not an `AutoModelForCausalLM`, gets an MTP draft
    from the MTP layers in its checkpoint (loading fails if any are missing)."""
    model_info = HF_EXAMPLE_MODELS.get_hf_info("GraniteDoclingForConditionalGeneration")
    model_info.check_transformers_version(on_fail="skip")
    from transformers.modeling_layers import MtpModel

    num_mtp_layers = 2
    # With fewer decoder than MTP layers, `MtpModel` would rename `mtp.layers` too
    sizes = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2)
    heads = dict(num_attention_heads=2, num_key_value_heads=2)
    text_config = dict(vocab_size=128, pad_token_id=0, num_mtp_layers=num_mtp_layers)
    config = AutoConfig.for_model(
        "granite_docling",
        text_config=sizes | heads | text_config,
        vision_config=sizes | heads | dict(out_hidden_size=64),
    )
    main_model = AutoModelForMultimodalLM.from_config(config)
    main_model.save_pretrained(tmp_path)
    # Granite Docling stores MTP layer `k` under `mtp.layers.{k}`
    weights = load_file(tmp_path / "model.safetensors")
    for name, param in MtpModel(main_model, num_mtp_layers).named_parameters():
        if name.startswith("layers."):
            weights[f"mtp.{name.replace('.mtp_block.', '.')}"] = param.detach()
    save_file(weights, tmp_path / "model.safetensors", {"format": "pt"})

    target = ModelConfig(str(tmp_path), model_impl="transformers", dtype="bfloat16")
    speculative_config = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=num_mtp_layers,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    draft = speculative_config.draft_model_config
    assert draft.architecture == "TransformersForMTP"
    vllm_config = VllmConfig(model_config=target, speculative_config=speculative_config)
    with set_current_vllm_config(vllm_config):
        model = model_loader.get_model(vllm_config=vllm_config, model_config=draft)
    assert len(model.model.layers) == num_mtp_layers


def test_init_parameters_keeps_declared_dtype_and_requires_grad():
    """Meta parameters materialize in their own dtype, as with `from_pretrained`,
    and an integer parameter, which cannot require grad, does not fail."""
    with torch.device("meta"):
        module = nn.Module()
        module.fp32 = nn.Parameter(torch.empty(2))
        module.fp8 = nn.Parameter(
            torch.empty(2, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        module.int8 = nn.Parameter(
            torch.empty(2, dtype=torch.int8), requires_grad=False
        )
    stub = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    Base.init_parameters(stub, module)

    assert module.fp32.dtype == torch.float32 and module.fp32.requires_grad
    assert module.fp8.dtype == torch.float8_e4m3fn and not module.fp8.requires_grad
    assert module.int8.dtype == torch.int8 and not module.int8.requires_grad


class _LoadPathStub(nn.Module):
    """A module with the `Base` methods that its weight loading calls."""

    _apply_conversion_mapping = Base._apply_conversion_mapping
    _drop_ignored_unexpected = Base._drop_ignored_unexpected


def _stub_for(model: nn.Module, *conversions) -> _LoadPathStub:
    """A stub whose `self.model` is `model`, loading with `conversions`."""
    stub = _LoadPathStub()
    stub.config = PreTrainedConfig()
    stub.model = model
    model.base_model_prefix = "model"
    if not hasattr(model, "_keys_to_ignore_on_load_unexpected"):
        model._keys_to_ignore_on_load_unexpected = None
    stub._weight_conversions = lambda: list(conversions)
    stub._maybe_apply_model_mapping = lambda: None
    stub.ignore_unexpected_prefixes = stub.ignore_unexpected_suffixes = []
    Base._create_hf_to_vllm_mapper(stub)
    return stub


def test_ignored_unexpected_patterns_match_renamed_keys():
    """Like Transformers, `_keys_to_ignore_on_load_unexpected` matches renamed keys
    and only drops weights nothing loads (A.X-K1's `post_mlp_layernorm` is renamed
    into `mlp` and unused only on its dense layers)."""
    model = nn.Module()
    model._keys_to_ignore_on_load_unexpected = [r"mlp\.post_norm"]
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.post_norm = nn.RMSNorm(2)
    stub = _stub_for(model, WeightRenaming("post_norm", "mlp.post_norm"))
    weights = [(f"model.layers.{i}.post_norm.weight", torch.ones(2)) for i in range(2)]

    loaded = Base.load_weights(stub, weights)

    assert loaded == {"model.layers.0.mlp.post_norm.weight"}


@pytest.mark.skipif(
    not hasattr(core_model_loading, "Split"), reason="`Split` needs Transformers 5.20"
)
def test_convert_weights_splits_fused_gqa_qkv(dist_init, default_vllm_config):
    """A checkpoint that fuses GQA `[q|k|v]` into one tensor (Sarvam-30B) is split
    into each projection's size, after the renamings, also into a quantized module
    unless it's stored unquantized (in another dtype than the module's weight). The
    fused tensor of a skipped module is left for the loader to drop."""
    model = _LoadPathStub()
    model.config = PreTrainedConfig()
    model.self_attn = nn.Module()
    model.self_attn.q_proj = ReplicatedLinear(8, 8, bias=False)
    model.self_attn.k_proj = ReplicatedLinear(8, 2, bias=False)
    model.self_attn.v_proj = ReplicatedLinear(8, 2, bias=False)
    model.vision_tower = StageMissingLayer("vision_tower")
    converter = WeightConverter(
        source_patterns="query_key_value",
        target_patterns=["q_proj", "k_proj", "v_proj"],
        operations=[core_model_loading.Split(dim=0)],
    )
    model.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_renaming=[
            WeightRenaming(r"^attention\.", "self_attn."),
            WeightRenaming(r"^vision_model\.", "vision_tower."),
        ]
    )
    model.weight_converters = [converter]
    fused = torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8)

    weights = model._apply_conversion_mapping
    # Cached by `id`, which a model freed by an earlier test may have had
    _model_to_pp_missing_layer_names.pop(id(model), None)
    fused_weight = ("attention.query_key_value.weight", fused)
    pieces = dict(weights([fused_weight]))

    assert list(pieces) == [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ]
    split = [pieces[f"self_attn.{p}_proj.weight"] for p in "qkv"]
    torch.testing.assert_close(torch.cat(split), fused)
    assert [p.shape[0] for p in split] == [8, 2, 2]

    model.self_attn.q_proj.quant_method = SimpleNamespace(uses_meta_device=False)
    assert len(list(weights([fused_weight]))) == 3
    unquantized = (fused_weight[0], fused.bfloat16())
    with pytest.raises(ValueError, match="split of a tensor stored quantized"):
        list(weights([unquantized]))

    skipped = ("vision_model.query_key_value.weight", fused)
    assert list(weights([skipped])) == [("vision_tower.q_proj.weight", fused)]


def test_converters_apply_to_renamed_weights():
    """Like in Transformers, converters see the renamed checkpoint names, so those
    scoped to a sub-model apply too (Molmo2's `ff_proj`), and a `Concatenate` joins
    all of its sources (its `wte` embeddings)."""
    model = nn.Module()
    model.language_model = text = nn.Module()
    text.embed_tokens = nn.Embedding(3, 2)
    text.up_proj = nn.Linear(2, 2, bias=False)
    text.gate_proj = nn.Linear(2, 2, bias=False)
    chunk = WeightConverter(
        source_patterns="ff_proj.weight",
        target_patterns=["up_proj.weight", "gate_proj.weight"],
        operations=[core_model_loading.Chunk(dim=0)],
    )
    concat = WeightConverter(
        source_patterns=["wte.embedding", "wte.new_embedding"],
        target_patterns="embed_tokens.weight",
        operations=[core_model_loading.Concatenate(dim=0)],
    )
    for converter in (chunk, concat):
        converter.scope_prefix, converter.base_model_prefix = "language_model", "model"
    renaming = WeightRenaming(r"transformer\.", "language_model.")
    stub = _stub_for(model, renaming, chunk, concat)
    ff_proj, wte = torch.rand(4, 2), [torch.rand(2, 2), torch.rand(1, 2)]
    checkpoint = {
        "model.transformer.ff_proj.weight": ff_proj,
        "model.transformer.wte.embedding": wte[0],
        "model.transformer.wte.new_embedding": wte[1],
    }

    Base.load_weights(stub, checkpoint.items())

    up_gate = torch.cat([text.up_proj.weight, text.gate_proj.weight])
    torch.testing.assert_close(up_gate, ff_proj)
    torch.testing.assert_close(text.embed_tokens.weight, torch.cat(wte))


@pytest.mark.skipif(
    not hasattr(core_model_loading, "Split"), reason="`Split` needs Transformers 5.20"
)
def test_split_into_vllm_fused_qkv(dist_init, default_vllm_config):
    """`Split` sizes targets that vLLM fused into a `QKVParallelLinear` by its full
    head counts (here GQA), and each piece loads into its shard (Molmo2's
    `att_proj`)."""
    model = nn.Module()
    model.self_attn = nn.Module()
    qkv = QKVParallelLinear(8, head_size=2, total_num_heads=4, total_num_kv_heads=1)
    model.self_attn.qkv_proj = qkv
    split = WeightConverter(
        source_patterns="query_key_value",
        target_patterns=["q_proj", "k_proj", "v_proj"],
        operations=[core_model_loading.Split(dim=0)],
    )
    stub = _stub_for(model, WeightRenaming(r"attention\.", "self_attn."), split)
    stub.hf_to_vllm_mapper |= WeightsMapper(
        orig_to_new_stacked={
            f"model.self_attn.{p}_proj": ("model.self_attn.qkv_proj", p) for p in "qkv"
        }
    )
    fused = torch.rand(12, 8)

    Base.load_weights(stub, [("model.attention.query_key_value.weight", fused)])

    torch.testing.assert_close(qkv.weight, fused)


def test_joining_into_a_quantized_module_is_rejected():
    """Converting into a quantized module by joining its sources (Molmo2's `wte`
    embeddings) is rejected, unless it quantizes online after loading."""
    model = nn.Module()
    model.embed_tokens = nn.Embedding(3, 2)
    concat = WeightConverter(
        source_patterns=["wte.embedding", "wte.new_embedding"],
        target_patterns="embed_tokens.weight",
        operations=[core_model_loading.Concatenate(dim=0)],
    )
    convert = _stub_for(model, concat)._apply_conversion_mapping
    wte = [("model.wte.embedding", torch.rand(2, 2))]
    wte += [("model.wte.new_embedding", torch.rand(1, 2))]

    model.embed_tokens.quant_method = SimpleNamespace(uses_meta_device=False)
    with pytest.raises(ValueError, match="split of a tensor stored quantized"):
        list(convert(wte))
    model.embed_tokens.quant_method.uses_meta_device = True
    assert [name for name, _ in convert(wte)] == ["model.embed_tokens.weight"]


def test_converted_weight_with_no_module_is_left_to_the_loader():
    """A converted weight that no module on this rank loads (e.g. an MTP layer the
    loader ignores) is passed on unconverted for the loader to drop or report."""
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    layer = model.layers[0]
    layer.up_proj = nn.Linear(2, 2, bias=False)
    layer.gate_proj = nn.Linear(2, 2, bias=False)
    chunk = WeightConverter(
        source_patterns="ff_proj.weight",
        target_patterns=["up_proj.weight", "gate_proj.weight"],
        operations=[core_model_loading.Chunk(dim=0)],
    )
    stub = _stub_for(model, chunk)
    stub.ignore_unexpected_prefixes = ["model.layers.1."]
    ff_proj = torch.rand(4, 2)
    checkpoint = [(f"model.layers.{i}.ff_proj.weight", ff_proj) for i in range(2)]

    Base.load_weights(stub, checkpoint)

    up_gate = torch.cat([layer.up_proj.weight, layer.gate_proj.weight])
    torch.testing.assert_close(up_gate, ff_proj)


def _quantized_split_stub() -> _LoadPathStub:
    """Molmo2's fused `att_proj` (a module's `Split`) and `ff_proj.weight` (a weight's
    `Chunk`), loaded into vLLM's fused `qkv_proj` and `gate_up_proj`."""
    model = _LoadPathStub()
    model.config = PreTrainedConfig()
    model.self_attn = nn.Module()
    model.self_attn.qkv_proj = QKVParallelLinear(
        6, head_size=2, total_num_heads=4, total_num_kv_heads=1
    )
    model.mlp = nn.Module()
    model.mlp.gate_up_proj = MergedColumnParallelLinear(6, [4, 4])
    stacked: dict[str, tuple[str, ShardId]] = {
        f"self_attn.{p}_proj": ("self_attn.qkv_proj", p) for p in "qkv"
    }
    stacked |= {"mlp.gate_proj": ("mlp.gate_up_proj", 0)}
    stacked |= {"mlp.up_proj": ("mlp.gate_up_proj", 1)}
    model.hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked=stacked)
    model.weight_converters = [
        WeightConverter(
            source_patterns="att_proj",
            target_patterns=["q_proj", "k_proj", "v_proj"],
            operations=[core_model_loading.Split(dim=0)],
        ),
        WeightConverter(
            source_patterns="ff_proj.weight",
            target_patterns=["up_proj.weight", "gate_proj.weight"],
            operations=[core_model_loading.Chunk(dim=0)],
        ),
    ]
    return model


@pytest.mark.skipif(
    not hasattr(core_model_loading, "Split"), reason="`Split` needs Transformers 5.20"
)
@pytest.mark.parametrize(
    ("name", "shape", "pieces"),
    [
        ("self_attn.att_proj.weight_scale", (1,), None),  # FP8 per tensor
        ("mlp.ff_proj.input_scale", (1,), None),
        ("self_attn.att_proj.input_scale", (), None),  # modelopt FP8
        ("mlp.ff_proj.weight_shape", (2,), None),  # int4 pack-quantized
        ("self_attn.att_proj.weight_scale", (12, 1), [(8, 1), (2, 1), (2, 1)]),
        ("mlp.ff_proj.weight_packed", (8, 3), [(4, 3), (4, 3)]),
        ("self_attn.att_proj.weight_scale", (6, 3), [(4, 3), (1, 3), (1, 3)]),
        ("mlp.ff_proj.weight_scale", (8,), [(4,), (4,)]),  # 1-D, the weight's rows
    ],
)
def test_convert_weights_splits_quantized_tensors(
    name, shape, pieces, dist_init, default_vllm_config
):
    """The other tensors a quantized checkpoint stores with a weight that a converter
    splits (Molmo2's and Sarvam-30B's quantized checkpoints) are split like the weight,
    by rows or blocks of rows (here per channel, packed and in blocks of 2), or copied
    to every piece (`None`) if they're a single value or 1-D of another length."""
    convert = _quantized_split_stub()._apply_conversion_mapping
    tensor = torch.rand(shape)
    module, param = name.rsplit(".", 1)
    prefix = module.rpartition(".")[0]
    targets = {"self_attn": ["q", "k", "v"], "mlp": ["up", "gate"]}[prefix]

    converted = dict(convert([(name, tensor)]))

    assert list(converted) == [f"{prefix}.{t}_proj.{param}" for t in targets]
    if pieces is None:
        assert all(piece is tensor for piece in converted.values())
    else:
        assert [tuple(piece.shape) for piece in converted.values()] == pieces
        torch.testing.assert_close(torch.cat(list(converted.values())), tensor)


@pytest.mark.skipif(
    not hasattr(core_model_loading, "Split"), reason="`Split` needs Transformers 5.20"
)
@pytest.mark.parametrize("module", ["self_attn.att_proj", "mlp.ff_proj"])
def test_convert_weights_rejects_quantized_tensors_that_dont_split(
    module, dist_init, default_vllm_config
):
    """A tensor of a split module that divides neither like the weight nor as a single
    value fails by name, for a module's converter and a weight's."""
    convert = _quantized_split_stub()._apply_conversion_mapping

    with pytest.raises(ValueError, match=f"Cannot split '{module}.weight_scale'"):
        list(convert([(f"{module}.weight_scale", torch.rand(5, 1))]))
