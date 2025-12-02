# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Set

import torch
from torch import nn
from transformers import BertConfig

from vllm.attention.layers.encoder_only_attention import EncoderOnlyAttention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, PoolerConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import (
    ClassifierPooler,
    DispatchPooler,
    Pooler,
    PoolingMethod,
    PoolingParamsUpdate,
    PoolingType,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces import SupportsCrossEncoding, SupportsQuant
from .interfaces_base import attn_type, default_pooling_type
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class BertEmbedding(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.position_embeddings = VocabParallelEmbedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
        )
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError(
                "Only 'absolute' position_embedding_type" + " is supported"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_type_ids = _decode_token_type_ids(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertPooler(Pooler):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.pooling = PoolingMethod.from_pooling_type(PoolingType.CLS)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooling.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.pooling.get_pooling_updates(task)

    def _head(self, pooled_output: torch.Tensor):
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor | list[torch.Tensor]:
        pooled_output = self.pooling(hidden_states, pooling_metadata)

        if isinstance(pooled_output, list):
            pooled_output = [self._head(output) for output in pooled_output]
        else:
            pooled_output = self._head(pooled_output)

        return pooled_output


class BertEncoder(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer = nn.ModuleList(
            [
                BertLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layer.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.attention = BertAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )

        self.intermediate = BertIntermediate(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.intermediate",
        )

        self.output = BertOutput(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
        )

    def forward(self, hidden_states: torch.Tensor):
        attn_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attn_output)
        output = self.output(intermediate_output, attn_output)
        return output


class BertAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.self = BertSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
        )

        self.output = BertSelfOutput(
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.output",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self_output = self.self(hidden_states)
        return self.output(self_output, hidden_states)


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_attention_heads
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == self.hidden_size

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.attn = EncoderOnlyAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        output = self.attn(q, k, v)
        return output


class BertSelfOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dense = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )
        self.intermediate_act_fn = get_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.dense = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@support_torch_compile
@default_pooling_type("CLS")
class BertModel(nn.Module, SupportsQuant):
    is_pooling_model = True

    packed_modules_mapping = {"qkv_proj": ["query", "key", "value"]}

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        embedding_class: type[nn.Module] = BertEmbedding,
    ) -> None:
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.embeddings = embedding_class(self.config)
        self.encoder = BertEncoder(vllm_config=vllm_config, prefix=f"{prefix}.encoder")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )

        return self.encoder(hidden_states)

    def _load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]

        loaded_stacked_params = []
        other_weights = []
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_params.append(name)
                break
            else:
                if name in params_dict:
                    other_weights.append((name, loaded_weight))

        return other_weights, loaded_stacked_params

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        other_weights, loaded_stacked_params = self._load_weights(weights)

        loader = AutoWeightsLoader(self, skip_prefixes=["pooler."])
        loaded_params = loader.load_weights(other_weights)
        loaded_params.update(loaded_stacked_params)
        return loaded_params


class BertPoolingModel(BertModel):
    is_pooling_model = True

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        embedding_class: type[nn.Module] = BertEmbedding,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            embedding_class=embedding_class,
        )

        config = vllm_config.model_config.hf_config
        self.pooler = BertPooler(config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        other_weights, loaded_stacked_params = self._load_weights(weights)

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(other_weights)
        loaded_params.update(loaded_stacked_params)
        return loaded_params


@default_pooling_type("CLS")
class BertEmbeddingModel(nn.Module, SupportsQuant):
    """A model that uses Bert to provide embedding functionalities.

    This class encapsulates the BertModel and provides an interface for
    embedding operations and customized pooling functions.

    Attributes:
        model: An instance of BertModel used for forward operations.
        _pooler: An instance of Pooler used for pooling operations.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.model = self._build_model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.pooler = self._build_pooler(pooler_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)

        has_model_prefix = any(name.startswith("model.") for name, _ in weights_list)
        if not has_model_prefix:
            mapper = WeightsMapper(orig_to_new_prefix={"": "model."})

        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."])
        return loader.load_weights(weights_list, mapper=mapper)

    def _build_model(self, vllm_config: VllmConfig, prefix: str = "") -> BertModel:
        return BertModel(
            vllm_config=vllm_config, prefix=prefix, embedding_class=BertEmbedding
        )

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        return DispatchPooler(
            {
                "token_embed": Pooler.for_token_embed(pooler_config),
                "embed": Pooler.for_embed(pooler_config),
            }
        )


# Here we encode the token type ids together with the input ids.
# Since we use int 32 for the input IDs and the vocabulary size
# is way lower than 2**31, there is room to encode additional
# bits. At the same time, for cross-encoder use cases, the
# token type ids are only 0 or 1, requiring only 1 bit.
# This means that we can store the token type ids in the 31st
# bit. We void the 32nd bit because that would produce a negative
# number, which could be used to signal other things.
#
# The reason for all of this is that all the tensors that are
# passed as input to the forward function of a module marked
# with @support_torch_compile have to be persistent. So to
# avoid adding more persistent tensors in the model runner, we
# encode more information in the same persistent tensor.
#
# Since the *ForClassification module is outside of the BertModel
# which is compiled, we can do the encoding here and then separate
# the information again in the Embedding  layer. Since with bit masks
# we can do this entirely with torch operations and without branching,
# it works with torch compile.

TOKEN_TYPE_SHIFT = 30


def _encode_token_type_ids(
    input_ids: torch.Tensor, token_type_ids: torch.Tensor
) -> None:
    # input_ids can be padded to the right
    input_ids[: token_type_ids.shape[0]].bitwise_or_(token_type_ids << TOKEN_TYPE_SHIFT)


def _decode_token_type_ids(input_ids: torch.Tensor) -> torch.Tensor:
    ids_mask = (
        torch.ones_like(input_ids, dtype=torch.int32, device=input_ids.device)
        << TOKEN_TYPE_SHIFT
    )
    tokens_mask = ids_mask.bitwise_not()

    token_type_ids = input_ids.bitwise_and(ids_mask) >> TOKEN_TYPE_SHIFT

    input_ids.bitwise_and_(tokens_mask)

    return token_type_ids


class BertMLMHead(nn.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

    def tie_weights_with_embeddings(self, embeddings_weight: torch.Tensor):
        self.decoder.weight = embeddings_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class SPLADESparsePooler(Pooler):
    """
    SPLADE sparse pooling:
    logits = mlm_head(hidden_states)
            -> log1p(relu(logits))
            -> (max|sum over L)
            -> [V]

    Padding is masked with an attention mask,
    [CLS]/[SEP] is removed (selected),
    and then pooled.
    """

    def __init__(
        self,
        mlm_head: nn.Module,
        cls_token_id: int | None = 101,
        sep_token_id: int | None = 102,
        pooling: str = "max",
        remove_cls_sep: bool = True,
    ):
        super().__init__()
        assert pooling in ("max", "sum")
        self.mlm_head = mlm_head
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pooling = pooling
        self.remove_cls_sep = remove_cls_sep

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        assert isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 2

        lens_tensor: torch.Tensor = pooling_metadata.prompt_lens
        lens: list[int] = lens_tensor.tolist()
        B: int = len(lens)

        token_ids = pooling_metadata.prompt_token_ids
        offset = 0
        pooled_list: list[torch.Tensor] = []

        for i in range(B):
            L = int(lens[i])
            hs = hidden_states[offset : offset + L]

            start_idx = 0
            end_idx = L
            if self.remove_cls_sep and token_ids is not None:
                if (
                    self.cls_token_id is not None
                    and token_ids[i, 0].item() == self.cls_token_id
                ):
                    start_idx = 1
                if (
                    self.sep_token_id is not None
                    and token_ids[i, L - 1].item() == self.sep_token_id
                ):
                    end_idx = max(start_idx, L - 1)

            if end_idx <= start_idx:
                V = int(self.mlm_head.decoder.out_features)
                pooled_list.append(hs.new_zeros((V,)))
                offset += L
                continue

            logits_i = self.mlm_head(hs[start_idx:end_idx])
            scores_i = torch.log1p(torch.relu(logits_i))

            if self.pooling == "sum":
                pooled_i = scores_i.sum(dim=0)
            else:  # "max"
                pooled_i = scores_i.max(dim=0).values

            pooled_list.append(pooled_i.contiguous())
            offset += L

        return torch.stack(pooled_list, dim=0).contiguous()


@default_pooling_type("CLS")
class BertSpladeSparseEmbeddingModel(BertEmbeddingModel):
    """
    BertEmbeddingModel + SPLADE sparse embedding.
    - Make logits by self.mlm_head
    - pooler: SPLADESparsePooler(mlm_head...)
    """

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", splade_pooling: str = "max"
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = vllm_config.model_config.hf_config

        # MLM head
        self.mlm_head = BertMLMHead(
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-12),
        )

        self._splade_pooling = splade_pooling
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = self._build_pooler(pooler_config)

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        cfg = self.model.config

        if not hasattr(self, "mlm_head"):
            self.mlm_head = BertMLMHead(
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
                layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-12),
            )

        pooling_mode = getattr(self, "_splade_pooling", "max")

        cls_id = getattr(cfg, "cls_token_id", None)
        sep_id = getattr(cfg, "sep_token_id", None)

        return DispatchPooler(
            {
                "token_embed": Pooler.for_token_embed(pooler_config),
                "embed": SPLADESparsePooler(
                    mlm_head=self.mlm_head,
                    cls_token_id=cls_id,
                    sep_token_id=sep_id,
                    pooling=pooling_mode,  # "max" or "sum"
                    remove_cls_sep=True,
                ),
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if not hasattr(self, "mlm_head"):
            cfg = self.model.config
            self.mlm_head = BertMLMHead(
                hidden_size=cfg.hidden_size,
                vocab_size=cfg.vocab_size,
                layer_norm_eps=getattr(cfg, "layer_norm_eps", 1e-12),
            )

        def _strip(name: str) -> str:
            for p in ("model.", "bert."):
                if name.startswith(p):
                    name = name[len(p) :]
            return name

        weights_list = list(weights)
        model_side: list[tuple[str, torch.Tensor]] = []
        mlm_side: list[tuple[str, torch.Tensor]] = []

        for k, w in weights_list:
            name = _strip(k)
            if name.startswith("cls.predictions."):
                mlm_side.append((name, w))
            else:
                model_side.append((name, w))

        loaded: set[str] = set()
        loaded_model = self.model.load_weights(model_side)
        loaded.update({"model." + n for n in loaded_model})

        if mlm_side:
            name_map = {
                "cls.predictions.transform.dense.weight": "mlm_head.dense.weight",
                "cls.predictions.transform.dense.bias": "mlm_head.dense.bias",
                ("cls.predictions.transform.LayerNorm.weight"): (
                    "mlm_head.layer_norm.weight"
                ),
                ("cls.predictions.transform.LayerNorm.bias"): (
                    "mlm_head.layer_norm.bias"
                ),
                "cls.predictions.decoder.weight": "mlm_head.decoder.weight",
                "cls.predictions.decoder.bias": "mlm_head.decoder.bias",
            }
            remapped = [(name_map[n], w) for n, w in mlm_side if n in name_map]
            if remapped:
                loaded_mlm = AutoWeightsLoader(self).load_weights(remapped)
                loaded.update(loaded_mlm)

        return loaded


@default_pooling_type("CLS")
class BertForSequenceClassification(nn.Module, SupportsCrossEncoding, SupportsQuant):
    """A model that uses Bert to provide embedding functionalities.

    This class encapsulates the BertModel and provides an interface for
    embedding operations and customized pooling functions.

    Attributes:
        model: An instance of BertModel used for forward operations.
        _pooler: An instance of Pooler used for pooling operations.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.num_labels = config.num_labels
        self.bert = BertPoolingModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "bert"),
            embedding_class=BertEmbedding,
        )
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels,
            dtype=vllm_config.model_config.head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {
                "token_classify": Pooler.for_token_classify(
                    pooler_config, classifier=self.classifier
                ),
                "classify": ClassifierPooler(
                    pooling=self.bert.pooler,
                    classifier=self.classifier,
                    act_fn="classify",
                ),
                "score": ClassifierPooler(
                    pooling=self.bert.pooler, classifier=self.classifier, act_fn="score"
                ),
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.bert.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights)
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is not None:
            assert self.bert.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
            assert input_ids is not None
            _encode_token_type_ids(input_ids, token_type_ids)

        return self.bert(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )


@attn_type("encoder_only")
@default_pooling_type("ALL")
class BertForTokenClassification(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.head_dtype = vllm_config.model_config.head_dtype
        self.num_labels = config.num_labels
        self.bert = BertModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "bert"),
            embedding_class=BertEmbedding,
        )
        self.classifier = nn.Linear(
            config.hidden_size, config.num_labels, dtype=self.head_dtype
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler(
            {
                "token_classify": Pooler.for_token_classify(
                    pooler_config=pooler_config
                ),
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.bert.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights)
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_type_ids is not None:
            assert self.bert.config.vocab_size < (1 << TOKEN_TYPE_SHIFT)
            assert input_ids is not None
            _encode_token_type_ids(input_ids, token_type_ids)

        hidden_states = self.bert(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

        hidden_states = hidden_states.to(self.head_dtype)
        return self.classifier(hidden_states)
