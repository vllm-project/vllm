# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
from collections.abc import Iterable
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

from .interfaces_base import VllmModelForPooling, is_pooling_model

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

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


def try_create_mm_pooling_model_cls(orig_cls: _T) -> _T:
    class CallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.calls = []

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                self.calls.append(node.func.id)
            self.generic_visit(node)

    visitor = CallVisitor()
    visitor.visit(ast.parse(inspect.getsource(orig_cls)))
    if "init_vllm_registered_model" not in visitor.calls:
        return None

    class ModelForPooling(orig_cls, VllmModelForPooling):
        is_pooling_model = True

        def __init__(
            self,
            *,
            vllm_config: "VllmConfig",
            prefix: str = "",
            **kwargs: Any,
        ) -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

            self.pooler = self.get_language_model().pooler

    return ModelForPooling  # type: ignore


def _create_pooling_model_cls(orig_cls: _T) -> _T:
    # Lazy import
    from .utils import AutoWeightsLoader, WeightsMapper

    class ModelForPooling(orig_cls, VllmModelForPooling):
        is_pooling_model = True

        def __init__(
            self,
            *,
            vllm_config: "VllmConfig",
            prefix: str = "",
            **kwargs: Any,
        ) -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

            self.vllm_config = vllm_config

            # These are not used in pooling models
            for attr in ("lm_head", "logits_processor"):
                if hasattr(self, attr):
                    delattr(self, attr)

            # If the model already defines a pooler instance, don't overwrite it
            if not getattr(self, "pooler", None):
                self._init_pooler(vllm_config, prefix=prefix)

        def _init_pooler(self, vllm_config: "VllmConfig", prefix: str = ""):
            raise NotImplementedError

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
            load_lm_head: bool = False,
        ):
            # TODO: Support uninitialized params tracking

            # For most pooling models: We have deleted this attribute, so don't load it.
            # For converting an LLM into a seq cls model, we need the lm_head.
            if not load_lm_head:
                weights = (
                    (name, data)
                    for name, data in weights
                    if not name.startswith("lm_head.")
                )

            # If `*ForCausalLM` defines `load_weights` on the inner model
            # and there are no other inner modules with parameters,
            # we support loading from both `*Model` and `*ForCausalLM`
            if hasattr(self, "model") and hasattr(self.model, "load_weights"):
                # Whether only `self.model` contains parameters
                model_is_only_param = all(
                    name == "model" or next(child.parameters(), None) is None
                    for name, child in self.named_children()
                )

                if model_is_only_param:
                    mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})
                    weights = mapper.apply(weights)

                    loaded_params = self.model.load_weights(weights)
                    loaded_params = {f"model.{name}" for name in loaded_params}
                    return loaded_params

            # For most other models
            if hasattr(orig_cls, "load_weights"):
                return orig_cls.load_weights(self, weights)  # type: ignore
            # Fallback
            else:
                loader = AutoWeightsLoader(self)
                return loader.load_weights(weights)

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
    from vllm.model_executor.layers.pooler import DispatchPooler, Pooler

    class ModelForEmbedding(_create_pooling_model_cls(cls)):
        def _init_pooler(self, vllm_config: "VllmConfig", prefix: str = ""):
            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            self.pooler = DispatchPooler(
                {
                    "token_embed": Pooler.for_token_embed(pooler_config),
                    "embed": Pooler.for_embed(pooler_config),
                },
            )

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
    from vllm.model_executor.layers.pooler import (
        DispatchPooler,
        Pooler,
    )
    from vllm.model_executor.models.interfaces import SupportsCrossEncoding

    from .utils import maybe_prefix

    class ModelForSequenceClassification(
        _create_pooling_model_cls(cls), SupportsCrossEncoding
    ):
        def _init_pooler(self, vllm_config: "VllmConfig", prefix: str = ""):
            text_config = vllm_config.model_config.hf_config.get_text_config()
            model_config = vllm_config.model_config
            quant_config = vllm_config.quant_config

            self.score = ReplicatedLinear(
                model_config.hidden_size,
                text_config.num_labels,
                bias=False,
                params_dtype=vllm_config.model_config.head_dtype,
                quant_config=quant_config,
                return_bias=False,
                prefix=maybe_prefix(prefix, "score"),
            )

            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            self.pooler = DispatchPooler(
                {
                    "token_classify": Pooler.for_token_classify(
                        pooler_config, classifier=self.score
                    ),
                    "classify": Pooler.for_classify(
                        pooler_config, classifier=self.score, act_fn="classify"
                    ),
                    "score": Pooler.for_classify(
                        pooler_config, classifier=self.score, act_fn="score"
                    ),
                }
            )

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            text_config = self.config.get_text_config()
            tokens = getattr(text_config, "classifier_from_token", None)
            method = getattr(text_config, "method", None)

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


def as_reward_model(cls: _T) -> _T:
    """
    Subclass an existing vLLM model to support reward modeling.

    By default, we return the hidden states of each token directly.

    Note:
        We assume that no extra layers are added to the original model;
        please implement your own model if this is not the case.
    """
    # Avoid modifying existing reward models
    if is_pooling_model(cls):
        return cls

    # Lazy import
    from vllm.model_executor.layers.pooler import DispatchPooler, Pooler

    from .interfaces_base import default_pooling_type

    @default_pooling_type("ALL")
    class ModelForReward(_create_pooling_model_cls(cls)):
        def _init_pooler(self, vllm_config: "VllmConfig", prefix: str = ""):
            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            self.pooler = DispatchPooler(
                {
                    "token_classify": Pooler.for_token_classify(
                        pooler_config=pooler_config
                    )
                }
            )

    ModelForReward.__name__ = _get_pooling_model_name(cls.__name__, "ForReward")

    return ModelForReward  # type: ignore


class SequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        text_config = vllm_config.model_config.hf_config.get_text_config()
        method = getattr(text_config, "method", None)
        tokens = getattr(text_config, "classifier_from_token", None)

        if method is None:
            return

        assert tokens is not None
        assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"

        if method == "from_2_way_softmax":
            assert len(tokens) == 2
            text_config.num_labels = 1
        else:
            text_config.num_labels = len(tokens)

        # `llm as reranker` defaults to not using pad_token
        use_pad_token = getattr(text_config, "use_pad_token", False)
        text_config.use_pad_token = use_pad_token


def load_weights_using_from_2_way_softmax(
    model, weights: Iterable[tuple[str, torch.Tensor]]
):
    # refer to https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    model_config = model.vllm_config.model_config
    quant_config = model.vllm_config.quant_config
    text_config = model.config.get_text_config()

    tokens = getattr(text_config, "classifier_from_token", [])
    tokens = cast(list[int], tokens)
    assert len(tokens) == 2

    model.lm_head = ParallelLMHead(
        text_config.vocab_size, text_config.hidden_size, quant_config=quant_config
    )
    if text_config.tie_word_embeddings:
        # embed_tokens is the assumed name for input embeddings. If the model does not
        # have this attribute, we fall back to get_input_embeddings(), which is used by
        # the Transformers modeling backend.
        embed_tokens = (
            model.model.embed_tokens
            if hasattr(model.model, "embed_tokens")
            else model.model.get_input_embeddings()
        )
        model.lm_head = model.lm_head.tie_weights(embed_tokens)

    # ModelForPooling is dynamically defined inside the _create_pooling_model_cls
    # function, so we need use this hacky method to obtain it.
    pooling_model_cls = next(
        x for x in type(model).__mro__ if x.__name__ == "ModelForPooling"
    )
    loaded_weights = pooling_model_cls.load_weights(model, weights, load_lm_head=True)

    from vllm.transformers_utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
    )

    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    score_weight = model.lm_head.weight.data[[true_id]].to(
        torch.float32
    ) - model.lm_head.weight.data[[false_id]].to(torch.float32)

    param = model.score.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, score_weight)

    del model.lm_head
    loaded_weights.add("score.weight")
    loaded_weights.discard("lm_head.weight")
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

    model.lm_head = ParallelLMHead(
        text_config.vocab_size, text_config.hidden_size, quant_config=quant_config
    )
    if text_config.tie_word_embeddings:
        # embed_tokens is the assumed name for input embeddings. If the model does not
        # have this attribute, we fall back to get_input_embeddings(), which is used by
        # the Transformers modeling backend.
        embed_tokens = (
            model.model.embed_tokens
            if hasattr(model.model, "embed_tokens")
            else model.model.get_input_embeddings()
        )
        model.lm_head = model.lm_head.tie_weights(embed_tokens)

    # Skip ModelForSequenceClassification in MRO to avoid infinite recursion
    loaded_weights = type(model).__mro__[1].load_weights(model, weights)

    from vllm.transformers_utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        revision=model_config.tokenizer_revision,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
    )

    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    score_weight = model.lm_head.weight.data[token_ids]

    param = model.score.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, score_weight)

    del model.lm_head
    loaded_weights.add("score.weight")
    loaded_weights.discard("lm_head.weight")
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

    text_config = model.vllm_config.model_config.hf_config.get_text_config()
    method = getattr(text_config, "method", None)
    assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"
    return SEQ_CLS_LOAD_METHODS[method](model, weights)
