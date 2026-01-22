# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.transformers_utils.config import (
    try_get_dense_modules,
)
from vllm.transformers_utils.repo_utils import get_hf_file_bytes

from .interfaces import supports_multimodal
from .interfaces_base import VllmModelForPooling, is_pooling_model

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.model_executor.layers.pooler import Pooler

_T = TypeVar("_T", bound=type[nn.Module])

logger = init_logger(__name__)

_GENERATE_SUFFIXES = [
    "ForCausalLM",
    "ForConditionalGeneration",
    "ChatModel",
    "LMHeadModel",
]


def _load_st_projector(model_config: "ModelConfig") -> nn.Module | None:
    """Load Sentence-Transformers Dense projection layers."""

    dense_modules = try_get_dense_modules(
        model_config.model, revision=model_config.revision
    )

    if dense_modules is None:
        return

    try:
        layers = []
        for layer_config in dense_modules:
            folder = layer_config["folder"]
            linear = nn.Linear(
                layer_config["in_features"],
                layer_config["out_features"],
                bias=layer_config.get("bias", True),
                dtype=model_config.head_dtype,
            )
            if not _load_dense_weights(linear, folder, model_config):
                continue
            layers.append(linear)
            if act_name := layer_config.get("activation_function"):
                layers.append(get_act_fn(act_name))
        return nn.Sequential(*layers).to(dtype=model_config.head_dtype)
    except Exception:
        logger.exception("ST projector loading failed")

    return None


def _load_dense_weights(
    linear: nn.Linear, folder: str, model_config: "ModelConfig"
) -> bool:
    """Load weights using vLLM's weight_loader pattern."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    for filename in ["model.safetensors", "pytorch_model.bin"]:
        file_path = f"{folder}/{filename}" if folder else filename

        try:
            file_bytes = get_hf_file_bytes(
                file_path, model_config.model, model_config.revision
            )
            if not file_bytes:
                continue

            if filename.endswith(".safetensors"):
                from safetensors.torch import load as load_safetensors

                state_dict = load_safetensors(file_bytes)
            else:
                import io

                state_dict = torch.load(
                    io.BytesIO(file_bytes), map_location="cpu", weights_only=True
                )

            for weight_key in ["weight", "linear.weight", "dense.weight"]:
                if weight_key in state_dict:
                    weight_loader = getattr(
                        linear.weight, "weight_loader", default_weight_loader
                    )
                    weight_loader(linear.weight, state_dict[weight_key])

                    bias_key = weight_key.replace("weight", "bias")
                    if linear.bias is not None and bias_key in state_dict:
                        bias_loader = getattr(
                            linear.bias, "weight_loader", default_weight_loader
                        )
                        bias_loader(linear.bias, state_dict[bias_key])
                    return True
        except Exception:
            logger.exception("Failed to load %s", filename)
            continue

    return False


def _get_pooling_model_name(orig_model_name: str, pooling_suffix: str) -> str:
    model_name = orig_model_name

    for generate_suffix in _GENERATE_SUFFIXES:
        model_name = model_name.removesuffix(generate_suffix)

    return model_name + pooling_suffix


def _create_pooling_model_cls(orig_cls: _T) -> _T:
    # Lazy import
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    from .utils import AutoWeightsLoader, StageMissingLayer, no_init_weights

    class ModelForPooling(orig_cls, VllmModelForPooling):
        is_pooling_model = True

        def __init__(
            self,
            *,
            vllm_config: "VllmConfig",
            prefix: str = "",
            **kwargs: Any,
        ) -> None:
            with no_init_weights(
                self,
                lambda mod: StageMissingLayer("output", mod),
                targets=(LogitsProcessor, ParallelLMHead),
            ):
                super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

            # Used by SEQ_CLS_LOAD_METHODS
            self.vllm_config = vllm_config

            # If the model already defines a pooler instance, don't overwrite it
            pooler = getattr(self, "pooler", None)
            if not pooler and supports_multimodal(self):
                # Try to get the pooler from the LM backbone
                language_model = self.get_language_model()
                if hasattr(language_model, "pooler"):
                    pooler = language_model.pooler

            if not pooler:
                pooler = self._init_pooler(vllm_config, prefix=prefix)

            self.pooler = pooler

        def _init_pooler(
            self,
            vllm_config: "VllmConfig",
            prefix: str = "",
        ) -> "Pooler":
            raise NotImplementedError

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            params_dict = dict(self.named_parameters())

            # We support loading from both `*ForCausalLM` and `*Model`
            candidate_prefixes = ["", "model."]
            target_prefix = ""

            seen_weights = list[tuple[str, torch.Tensor]]()
            for name, loaded_weight in weights:
                seen_weights.append((name, loaded_weight))

                try:
                    target_prefix = next(
                        prefix
                        for prefix in candidate_prefixes
                        if prefix + name in params_dict
                    )
                    break
                except StopIteration:
                    # The weight might not exist on the model
                    # (to be handled by AutoWeightsLoader)
                    pass

            if target_prefix:
                target_model = self
                for attr in target_prefix.split("."):
                    if attr:
                        target_model = getattr(self, attr)

                logger.info(
                    "Mapping weights to %s as they are "
                    "relative to this model instead of %s.",
                    target_model._get_name(),
                    self._get_name(),
                )

            mapped_weights = (
                (target_prefix + name, weight)
                for name, weight in (*seen_weights, *weights)
            )

            def default_load_weights(weights):
                loader = AutoWeightsLoader(self)
                return loader.load_weights(weights)

            load_weights = getattr(super(), "load_weights", default_load_weights)
            return load_weights(mapped_weights)

    return ModelForPooling  # type: ignore


def as_embedding_model(cls: _T) -> _T:
    """
    Subclass an existing vLLM model to support embeddings.

    By default, the embeddings of the whole prompt are extracted from the
    normalized hidden state corresponding to the last token.

    Note:
        We assume that no extra layers are added to the original model;
        please implement your own model if this is not the case.
    """
    # Avoid modifying existing embedding models
    if is_pooling_model(cls):
        return cls

    # Lazy import
    from vllm.model_executor.layers.pooler import DispatchPooler

    class ModelForEmbedding(_create_pooling_model_cls(cls)):
        def _init_pooler(
            self,
            vllm_config: "VllmConfig",
            prefix: str = "",
        ) -> "Pooler":
            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            return DispatchPooler.for_embedding(pooler_config)

    ModelForEmbedding.__name__ = _get_pooling_model_name(cls.__name__, "ForEmbedding")

    return ModelForEmbedding  # type: ignore


def as_seq_cls_model(cls: _T) -> _T:
    """
    Subclass an existing vLLM model to support classify and score tasks.

    By default, the class probabilities are extracted from the softmaxed
    hidden state corresponding to the last token.

    Note:
        We assume that the classification head is a single linear layer
        stored as the attribute `score` of the top-level model;
        please implement your own model if this is not the case.
    """
    # Avoid modifying existing classification models
    if is_pooling_model(cls):
        return cls

    # Lazy import
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.pooler import DispatchPooler
    from vllm.model_executor.models.interfaces import SupportsCrossEncoding

    from .utils import maybe_prefix

    class ModelForSequenceClassification(
        _create_pooling_model_cls(cls), SupportsCrossEncoding
    ):
        def _init_pooler(
            self,
            vllm_config: "VllmConfig",
            prefix: str = "",
        ) -> "Pooler":
            text_config = vllm_config.model_config.hf_config.get_text_config()
            model_config = vllm_config.model_config
            quant_config = vllm_config.quant_config

            self.score = ReplicatedLinear(
                model_config.get_hidden_size(),
                text_config.num_labels,
                bias=False,
                params_dtype=vllm_config.model_config.head_dtype,
                quant_config=quant_config,
                return_bias=False,
                prefix=maybe_prefix(prefix, "score"),
            )

            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            return DispatchPooler.for_seq_cls(pooler_config, classifier=self.score)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            hf_config = self.config
            text_config = hf_config.get_text_config()
            tokens = getattr(
                hf_config,
                "classifier_from_token",
                getattr(text_config, "classifier_from_token", None),
            )
            method = getattr(hf_config, "method", getattr(text_config, "method", None))

            def auto_set_score_bias(weights):
                for name, weight in weights:
                    if name == "score.bias":
                        device = self.score.weight.device
                        dtype = self.score.weight.dtype
                        bias = weight.to(device).to(dtype)
                        self.score.bias = torch.nn.Parameter(bias)
                        self.score.skip_bias_add = False
                    else:
                        yield name, weight

            weights = auto_set_score_bias(weights)
            if tokens is None and method is None:
                return super().load_weights(weights)
            else:
                # Online convert ForCausalLM into
                # ForSequenceClassification model.
                return seq_cls_model_loader(self, weights)

    ModelForSequenceClassification.__name__ = _get_pooling_model_name(
        cls.__name__, "ForSequenceClassification"
    )

    return ModelForSequenceClassification  # type: ignore


class SequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        hf_config = vllm_config.model_config.hf_config
        text_config = hf_config.get_text_config()
        method = getattr(hf_config, "method", getattr(text_config, "method", None))
        tokens = getattr(
            hf_config,
            "classifier_from_token",
            getattr(text_config, "classifier_from_token", None),
        )

        if method is None:
            return

        assert tokens is not None
        assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"

        if method == "from_2_way_softmax":
            assert len(tokens) == 2
            hf_config.num_labels = 1
            text_config.num_labels = 1
        else:
            hf_config.num_labels = len(tokens)
            text_config.num_labels = len(tokens)

        # `llm as reranker` defaults to not using separating token.
        use_sep_token = getattr(text_config, "use_sep_token", False)
        text_config.use_sep_token = use_sep_token


def _get_language_model_for_seq_cls(model) -> nn.Module:
    """
    Get the language model component for sequence classification conversion.
    For VLMs, returns the inner language model. For standard LLMs, returns model itself.
    """
    if supports_multimodal(model):
        try:
            lm = model.get_language_model()
            if lm is not model:
                return lm
        except Exception:
            pass

    for attr_name in ("language_model", "lm", "text_model"):
        if hasattr(model, attr_name):
            candidate = getattr(model, attr_name)
            if (
                isinstance(candidate, nn.Module)
                and candidate is not model
                and hasattr(candidate, "model")
            ):
                return candidate

    for name, child in model.named_children():
        child_name = type(child).__name__
        if ("ForCausalLM" in child_name or "LMHead" in child_name) and hasattr(
            child, "model"
        ):
            return child

    return model


@contextmanager
def _disable_seq_cls_loading_on_inner_model(language_model, is_vlm: bool):
    """
    Context manager to temporarily disable sequence classification loading
    on inner VLM models to prevent recursive seq_cls_model_loader calls.
    """
    if not is_vlm:
        yield
        return

    inner_hf_config = getattr(language_model, "config", None)
    if inner_hf_config is None:
        yield
        return

    inner_text_config = inner_hf_config.get_text_config()
    original_method = getattr(inner_text_config, "method", None)
    original_tokens = getattr(inner_text_config, "classifier_from_token", None)
    original_hf_tokens = getattr(inner_hf_config, "classifier_from_token", None)

    try:
        if original_method is not None:
            inner_text_config.method = None
        if original_tokens is not None:
            inner_text_config.classifier_from_token = None
        if original_hf_tokens is not None:
            inner_hf_config.classifier_from_token = None
        yield
    finally:
        if original_method is not None:
            inner_text_config.method = original_method
        if original_tokens is not None:
            inner_text_config.classifier_from_token = original_tokens
        if original_hf_tokens is not None:
            inner_hf_config.classifier_from_token = original_hf_tokens


def load_weights_using_from_2_way_softmax(
    model, weights: Iterable[tuple[str, torch.Tensor]]
):
    # refer to https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    model_config = model.vllm_config.model_config
    quant_config = model.vllm_config.quant_config
    hf_config = model.config
    text_config = hf_config.get_text_config()

    tokens = getattr(
        hf_config,
        "classifier_from_token",
        getattr(text_config, "classifier_from_token", []),
    )
    tokens = cast(list[int], tokens)
    assert len(tokens) == 2

    language_model = _get_language_model_for_seq_cls(model)
    is_vlm = language_model is not model
    using_vlm_head = is_vlm and hasattr(language_model, "score")

    language_model.lm_head = ParallelLMHead(
        text_config.vocab_size, text_config.hidden_size, quant_config=quant_config
    )
    if text_config.tie_word_embeddings:
        # embed_tokens is the assumed name for input embeddings. If the model does not
        # have this attribute, we fall back to get_input_embeddings(), which is used by
        # the Transformers modeling backend.
        text_backbone = language_model.model
        embed_tokens = (
            text_backbone.embed_tokens
            if hasattr(text_backbone, "embed_tokens")
            else text_backbone.get_input_embeddings()
        )
        language_model.lm_head = language_model.lm_head.tie_weights(embed_tokens)

    # ModelForPooling is dynamically defined inside the _create_pooling_model_cls
    # function, so we need use this hacky method to obtain it.
    pooling_model_cls = next(
        x for x in type(model).__mro__ if x.__name__ == "ModelForPooling"
    )
    loaded_weights = pooling_model_cls.load_weights(model, weights)

    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
    )

    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    lm_head_weight = language_model.lm_head.weight
    score_weight = lm_head_weight.data[[true_id]].to(
        torch.float32
    ) - lm_head_weight.data[[false_id]].to(torch.float32)

    score_layer = language_model.score if using_vlm_head else model.score
    param = score_layer.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, score_weight)

    del language_model.lm_head

    score_weight_name = (
        "language_model.score.weight" if using_vlm_head else "score.weight"
    )
    loaded_weights.add(score_weight_name)

    lm_head_name = "lm_head.weight"
    if hf_to_vllm_mapper := getattr(model, "hf_to_vllm_mapper", None):
        lm_head_name = hf_to_vllm_mapper._map_name(lm_head_name)
    loaded_weights.discard(lm_head_name)
    return loaded_weights


def load_weights_no_post_processing(model, weights: Iterable[tuple[str, torch.Tensor]]):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    model_config = model.vllm_config.model_config
    quant_config = model.vllm_config.quant_config
    text_config = model.config.get_text_config()

    tokens = getattr(text_config, "classifier_from_token", [])
    tokens = cast(list[int], tokens)
    assert len(tokens) > 0

    language_model = _get_language_model_for_seq_cls(model)
    is_vlm = language_model is not model
    using_vlm_head = is_vlm and hasattr(language_model, "score")

    language_model.lm_head = ParallelLMHead(
        text_config.vocab_size, text_config.hidden_size, quant_config=quant_config
    )
    if text_config.tie_word_embeddings:
        # embed_tokens is the assumed name for input embeddings. If the model does not
        # have this attribute, we fall back to get_input_embeddings(), which is used by
        # the Transformers modeling backend.
        text_backbone = language_model.model
        embed_tokens = (
            text_backbone.embed_tokens
            if hasattr(text_backbone, "embed_tokens")
            else text_backbone.get_input_embeddings()
        )
        language_model.lm_head = language_model.lm_head.tie_weights(embed_tokens)

    with _disable_seq_cls_loading_on_inner_model(language_model, is_vlm):
        pooling_model_cls = next(
            x for x in type(model).__mro__ if x.__name__ == "ModelForPooling"
        )
        # Skip ModelForSequenceClassification in MRO to avoid infinite recursion
        loaded_weights = pooling_model_cls.load_weights(model, weights)

    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
    )

    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    score_weight = language_model.lm_head.weight.data[token_ids]

    score_layer = language_model.score if using_vlm_head else model.score
    param = score_layer.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, score_weight)

    del language_model.lm_head

    score_weight_name = (
        "language_model.score.weight" if using_vlm_head else "score.weight"
    )
    loaded_weights.add(score_weight_name)

    lm_head_name = "lm_head.weight"
    if hf_to_vllm_mapper := getattr(model, "hf_to_vllm_mapper", None):
        lm_head_name = hf_to_vllm_mapper._map_name(lm_head_name)
    loaded_weights.discard(lm_head_name)
    return loaded_weights


SEQ_CLS_LOAD_METHODS = {
    "from_2_way_softmax": load_weights_using_from_2_way_softmax,
    "no_post_processing": load_weights_no_post_processing,
}


def seq_cls_model_loader(model, weights: Iterable[tuple[str, torch.Tensor]]):
    # Online convert ForCausalLM into ForSequenceClassification model.
    # - from_2_way_softmax:
    #   - Qwen3ForCausalLM
    #     - Qwen3-Reranker
    #   - Qwen2ForCausalLM
    #     - mxbai-rerank-v2
    # - no_post_processing:
    #   - GemmaForCausalLM
    #     - bge-reranker-v2-gemma

    hf_config = model.vllm_config.model_config.hf_config
    text_config = hf_config.get_text_config()
    method = getattr(hf_config, "method", getattr(text_config, "method", None))
    assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"
    return SEQ_CLS_LOAD_METHODS[method](model, weights)
