# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.transformers_utils.config import (get_hf_file_bytes,
                                            get_hf_file_to_dict)

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


def _load_st_projector(model_config: "ModelConfig") -> Optional[nn.Module]:
    """Load Sentence-Transformers Dense projection layers."""

    try:
        modules = get_hf_file_to_dict("modules.json", model_config.model,
                                      model_config.revision)
        if not modules:
            return None

        if isinstance(modules, dict):
            modules = modules.get("modules", [])

        dense_modules = [
            m for m in modules
            if m.get("type") == "sentence_transformers.models.Dense"
        ]
        if not dense_modules:
            return None

        layers = []
        for module in dense_modules:
            folder = module.get("path", "")

            config_path = f"{folder}/config.json" if folder else "config.json"
            layer_config = get_hf_file_to_dict(config_path, model_config.model,
                                               model_config.revision)
            if not layer_config:
                continue

            linear = nn.Linear(layer_config.get("in_features", 768),
                               layer_config.get("out_features", 768),
                               bias=layer_config.get("bias", True),
                               dtype=model_config.head_dtype)

            if not _load_dense_weights(linear, folder, model_config):
                continue

            layers.append(linear)
            if act_name := layer_config.get("activation_function"):
                layers.append(get_act_fn(act_name))
        return nn.Sequential(*layers).to(dtype=model_config.head_dtype)
    except Exception:
        logger.exception("ST projector loading failed")

    return None


def _load_dense_weights(linear: nn.Linear, folder: str,
                        model_config: "ModelConfig") -> bool:
    """Load weights using vLLM's weight_loader pattern."""
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader)

    for filename in ["model.safetensors", "pytorch_model.bin"]:
        file_path = f"{folder}/{filename}" if folder else filename

        try:
            file_bytes = get_hf_file_bytes(file_path, model_config.model,
                                           model_config.revision)
            if not file_bytes:
                continue

            if filename.endswith(".safetensors"):
                from safetensors.torch import load as load_safetensors
                state_dict = load_safetensors(file_bytes)
            else:
                import io
                state_dict = torch.load(io.BytesIO(file_bytes),
                                        map_location="cpu",
                                        weights_only=True)

            for weight_key in ["weight", "linear.weight", "dense.weight"]:
                if weight_key in state_dict:
                    weight_loader = getattr(linear.weight, "weight_loader",
                                            default_weight_loader)
                    weight_loader(linear.weight, state_dict[weight_key])

                    bias_key = weight_key.replace("weight", "bias")
                    if linear.bias is not None and bias_key in state_dict:
                        bias_loader = getattr(linear.bias, "weight_loader",
                                              default_weight_loader)
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

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            # TODO: Support uninitialized params tracking

            # We have deleted this attribute, so don't load it
            weights = ((name, data) for name, data in weights
                       if not name.startswith("lm_head."))

            # If `*ForCausalLM` defines `load_weights` on the inner model
            # and there are no other inner modules with parameters,
            # we support loading from both `*Model` and `*ForCausalLM`
            if hasattr(self, "model") and hasattr(self.model, "load_weights"):
                # Whether only `self.model` contains parameters
                model_is_only_param = all(
                    name == "model" or next(child.parameters(), None) is None
                    for name, child in self.named_children())

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
                    "encode": Pooler.for_encode(pooler_config),
                    "embed": Pooler.for_embed(pooler_config),
                }, )

    ModelForEmbedding.__name__ = \
        _get_pooling_model_name(cls.__name__, "ForEmbedding")

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
    from vllm.model_executor.layers.pooler import (ClassifierPooler,
                                                   DispatchPooler, Pooler,
                                                   PoolingMethod, PoolingType)
    from vllm.model_executor.models.interfaces import SupportsCrossEncoding
    from vllm.sequence import IntermediateTensors

    from .utils import get_model_hidden_size, maybe_prefix

    class ModelForSequenceClassification(_create_pooling_model_cls(cls),
                                         SupportsCrossEncoding):

        def _init_pooler(self, vllm_config: "VllmConfig", prefix: str = ""):
            config = vllm_config.model_config.hf_config
            quant_config = vllm_config.quant_config
            hidden_size = get_model_hidden_size(config)

            self.score = ReplicatedLinear(
                hidden_size,
                config.num_labels,
                bias=False,
                params_dtype=torch.float32,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "score"),
            )

            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            pooling_type_str = pooler_config.pooling_type
            assert pooling_type_str is not None
            pooling_type = PoolingType[pooling_type_str]

            self.pooler = DispatchPooler({
                "encode":
                Pooler.for_encode(pooler_config),
                "classify":
                ClassifierPooler(
                    pooling=PoolingMethod.from_pooling_type(pooling_type),
                    classifier=self._classifier,
                    act_fn=ClassifierPooler.act_fn_for_seq_cls(
                        vllm_config.model_config),
                ),
                "score":
                ClassifierPooler(
                    pooling=PoolingMethod.from_pooling_type(pooling_type),
                    classifier=self._classifier,
                    act_fn=ClassifierPooler.act_fn_for_cross_encoder(
                        vllm_config.model_config),
                ),
            })

        def _classifier(self, x: torch.Tensor):
            x, _ = self.score(x.float())
            return x

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return super().forward(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            tokens = getattr(self.config, "classifier_from_token", None)
            method = getattr(self.config, "method", None)

            if tokens is None and method is None:
                return super().load_weights(weights)
            else:
                # Online convert ForCausalLM into
                # ForSequenceClassification model.
                return seq_cls_model_loader(self, weights)


    ModelForSequenceClassification.__name__ = \
        _get_pooling_model_name(cls.__name__, "ForSequenceClassification")

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

    class ModelForReward(_create_pooling_model_cls(cls)):

        def _init_pooler(self, vllm_config: "VllmConfig", prefix: str = ""):
            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None

            self.pooler = DispatchPooler(
                {"encode": Pooler.for_encode(pooler_config)}, )

    ModelForReward.__name__ = \
        _get_pooling_model_name(cls.__name__, "ForReward")

    return ModelForReward  # type: ignore


class SequenceClassificationConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config
        method = getattr(config, "method", None)
        tokens = getattr(config, "classifier_from_token", None)

        if method is None:
            return

        assert tokens is not None
        assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"

        if method == "from_2_way_softmax":
            assert len(tokens) == 2
            config.num_labels = 1
        else:
            config.num_labels = len(tokens)

        # `llm as reranker` defaults to not using pad_token
        use_pad_token = getattr(config, "use_pad_token", False)
        config.use_pad_token = use_pad_token


def load_weights_using_from_2_way_softmax(
        model, weights: Iterable[tuple[str, torch.Tensor]]):
    # refer to https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead)
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader)
    from vllm.model_executor.models.utils import AutoWeightsLoader

    model_config = model.vllm_config.model_config

    tokens = getattr(model.config, "classifier_from_token", [])
    tokens = cast(list[int], tokens)
    assert len(tokens) == 2

    if model.config.tie_word_embeddings:
        model.lm_head = model.model.embed_tokens
    else:
        quant_config = model.vllm_config.quant_config
        model.lm_head = ParallelLMHead(model.config.vocab_size,
                                       model.config.hidden_size,
                                       quant_config=quant_config)

    loader = AutoWeightsLoader(model)
    loaded_weights = loader.load_weights(weights)

    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(model_config.tokenizer,
                              revision=model_config.tokenizer_revision,
                              tokenizer_mode=model_config.tokenizer_mode,
                              trust_remote_code=model_config.trust_remote_code)

    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    score_weight = model.lm_head.weight.data[[true_id]].to(
        torch.float32) - model.lm_head.weight.data[[false_id]].to(
            torch.float32)

    param = model.score.weight
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, score_weight)

    del model.lm_head
    loaded_weights.add("score.weight")
    loaded_weights.discard("lm_head.weight")
    return loaded_weights


def load_weights_no_post_processing(model,
                                    weights: Iterable[tuple[str,
                                                            torch.Tensor]]):
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead)
    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader)
    from vllm.model_executor.models.utils import AutoWeightsLoader

    model_config = model.vllm_config.model_config
    tokens = getattr(model.config, "classifier_from_token", [])
    tokens = cast(list[int], tokens)
    assert len(tokens) > 0

    if model.config.tie_word_embeddings:
        model.lm_head = model.model.embed_tokens
    else:
        quant_config = model.vllm_config.quant_config
        model.lm_head = ParallelLMHead(model.config.vocab_size,
                                       model.config.hidden_size,
                                       quant_config=quant_config)

    loader = AutoWeightsLoader(model)
    loaded_weights = loader.load_weights(weights)

    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(model_config.tokenizer,
                              revision=model_config.tokenizer_revision,
                              tokenizer_mode=model_config.tokenizer_mode,
                              trust_remote_code=model_config.trust_remote_code)

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

    config = model.vllm_config.model_config.hf_config
    method = getattr(config, "method", None)
    assert method in SEQ_CLS_LOAD_METHODS, f"method {method} not supported"
    return SEQ_CLS_LOAD_METHODS[method](model, weights)
