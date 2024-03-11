# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertConfig, PretrainedConfig
from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertForPreTrainingOutput,
)

from flash_attn.bert_padding import (
    index_first_axis,
    index_first_axis_residual,
    pad_input,
    unpad_input,
)
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import BertEmbeddings
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import FusedMLP, Mlp
from flash_attn.utils.pretrained import state_dict_from_pretrained

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:
    layer_norm_fn = None


try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None


logger = logging.getLogger(__name__)


def create_mixer_cls(config, cross_attn=False, return_residual=False):
    use_flash_attn = getattr(config, "use_flash_attn", False)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    rotary_kwargs = {}
    if config.position_embedding_type == "rotary":
        rotary_kwargs["rotary_emb_dim"] = getattr(config, "rotary_emb_dim", config.hidden_size)
        rotary_kwargs["rotary_emb_base"] = getattr(config, "rotary_emb_base", 10000.0)
        rotary_kwargs["rotary_emb_scale_base"] = getattr(config, "rotary_emb_scale_base", None)
        rotary_kwargs["rotary_emb_interleaved"] = getattr(config, "rotary_emb_interleaved", False)
    mixer_cls = partial(
        MHA,
        num_heads=config.num_attention_heads,
        cross_attn=cross_attn,
        dropout=config.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=fused_bias_fc,
        use_flash_attn=use_flash_attn,
        return_residual=return_residual,
        **rotary_kwargs,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, return_residual=False):
    inner_dim = config.intermediate_size
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"], (
            "fused_mlp only " "supports approximate gelu"
        )
    if not fused_mlp:
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate=approximate),
            return_residual=return_residual,
        )
    else:
        if FusedMLP is None:
            raise ImportError("fused_dense is not installed")
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(
            FusedMLP,
            hidden_features=inner_dim,
            checkpoint_lvl=mlp_checkpoint_lvl,
            return_residual=return_residual,
        )
    return mlp_cls


def create_block(config, layer_idx=None):
    last_layer_subset = getattr(config, "last_layer_subset", False)
    cross_attn = last_layer_subset and layer_idx == config.num_hidden_layers - 1
    # TD [2022-12-19]: For cross attention (last layer), we actually want to return the
    # residual x_kv, not residual x. But it's annoying to change the API (and it only affects
    # one layer) so we just choose not to return residual in this case.
    return_residual = not cross_attn
    mixer_cls = create_mixer_cls(config, cross_attn, return_residual=return_residual)
    mlp_cls = create_mlp_cls(config, layer_idx, return_residual=return_residual)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_eps)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=False,
        resid_dropout1=config.hidden_dropout_prob,
        resid_dropout2=config.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        return_residual=return_residual,
    )
    return block


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.use_flash_attn = getattr(config, "use_flash_attn", False)
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, key_padding_mask=None, subset_mask=None):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = (
                {"key_padding_mask": key_padding_mask} if key_padding_mask is not None else None
            )
            for layer in self.layers:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, key_padding_mask
            )
            mixer_kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen_in_batch}
            if subset_mask is None:
                for layer in self.layers:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            else:
                for layer in self.layers[:-1]:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                if key_padding_mask is not None:
                    subset_idx = torch.nonzero(
                        subset_mask[key_padding_mask], as_tuple=False
                    ).flatten()
                    subset_seqlens = (subset_mask & key_padding_mask).sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                else:
                    subset_idx = torch.nonzero(subset_mask, as_tuple=False).flatten()
                    subset_seqlens = subset_mask.sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32), (1, 0)
                    )
                hidden_states_subset, hidden_states = index_first_axis_residual(
                    hidden_states, subset_idx
                )
                # It's ok to set max_seqlen_q to be much larger
                mixer_kwargs = {
                    "x_kv": hidden_states,
                    "cu_seqlens": subset_cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_k": max_seqlen_in_batch,
                }
                hidden_states = self.layers[-1](hidden_states_subset, mixer_kwargs=mixer_kwargs)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        self.transform_act_fn = nn.GELU(approximate=approximate)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.layer_norm.weight, self.layer_norm.bias, eps=self.layer_norm.eps
            )
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense

        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = linear_cls(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name, config, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPretraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        load_return = model.load_state_dict(
            remap_state_dict(state_dict_from_pretrained(model_name), config), strict=False
        )
        logger.info(load_return)
        return model


class BertModel(BertPreTrainedModel):
    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += self.pad_vocab_size_multiple - (
                config.vocab_size % self.pad_vocab_size_multiple
            )
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")
        assert config.hidden_act in ["gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"]

        self.embeddings = BertEmbeddings(
            config.hidden_size,
            config.vocab_size,
            config.max_position_embeddings,
            config.type_vocab_size,
            padding_idx=config.pad_token_id,
        )
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens_mask=None,
    ):
        """If masked_tokens_mask is not None (i.e. last_layer_subset == True in BertForPreTraining),
        we only want the output for the masked tokens. This means that we only compute the last
        layer output for these tokens.
        masked_tokens_mask: (batch, seqlen), dtype=torch.bool
        """
        hidden_states = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        # TD [2022-12:18]: Don't need to force residual in fp32
        # BERT puts embedding LayerNorm before embedding dropout.
        if not self.fused_dropout_add_ln:
            hidden_states = self.emb_ln(hidden_states)
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.emb_ln.weight, self.emb_ln.bias, eps=self.emb_ln.eps
            )
        hidden_states = self.emb_drop(hidden_states)

        if masked_tokens_mask is not None:
            batch_size, seqlen = input_ids.shape[:2]
            # We also need the first column for the CLS token
            first_col_mask = torch.zeros(
                batch_size, seqlen, dtype=torch.bool, device=input_ids.device
            )
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None

        sequence_output = self.encoder(
            hidden_states, key_padding_mask=attention_mask, subset_mask=subset_mask
        )

        if masked_tokens_mask is None:
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            if attention_mask is not None:
                subset_idx = subset_mask[attention_mask]
                pool_input = sequence_output[first_col_mask[attention_mask][subset_idx]]
                sequence_output = sequence_output[masked_tokens_mask[attention_mask][subset_idx]]
            else:
                pool_input = sequence_output[first_col_mask[subset_mask]]
                sequence_output = sequence_output[masked_tokens_mask[subset_mask]]
            pooled_output = self.pooler(pool_input, pool=False) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        # If dense_seq_output, we only need to pass the hidden states for the masked out tokens
        # (around 15%) to the classifier heads.
        self.dense_seq_output = getattr(config, "dense_seq_output", False)
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, "last_layer_subset", False)
        if self.last_layer_subset:
            assert self.dense_seq_output, "last_layer_subset requires dense_seq_output"
        use_xentropy = getattr(config, "use_xentropy", False)
        if use_xentropy and CrossEntropyLoss is None:
            raise ImportError("xentropy_cuda is not installed")
        loss_cls = (
            nn.CrossEntropyLoss
            if not use_xentropy
            else partial(CrossEntropyLoss, inplace_backward=True)
        )

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.mlm_loss = loss_cls(ignore_index=0)
        self.nsp_loss = loss_cls(ignore_index=-1)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, initializer_range=config.initializer_range))
        self.tie_weights()

    def tie_weights(self):
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        next_sentence_label=None,
    ):
        """
        If labels are provided, they must be 0 for masked out tokens (as specified in the attention
        mask).
        Outputs:
            if `labels` and `next_sentence_label` are not `None`:
                Outputs the total_loss which is the sum of the masked language modeling loss and the next
                sentence classification loss.
            if `labels` or `next_sentence_label` is `None`:
                Outputs a tuple comprising
                - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
                - the next sentence classification logits of shape [batch_size, 2].

        """
        masked_tokens_mask = labels > 0 if (self.last_layer_subset and labels is not None) else None
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            masked_tokens_mask=masked_tokens_mask,
        )
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        if self.dense_seq_output and labels is not None:
            masked_token_idx = torch.nonzero(labels.flatten() > 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(
                    rearrange(sequence_output, "b s d -> (b s) d"), masked_token_idx
                )
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            if (
                self.dense_seq_output and labels is not None
            ):  # prediction_scores are already flattened
                masked_lm_loss = self.mlm_loss(
                    prediction_scores, labels.flatten()[masked_token_idx]
                )
            else:
                masked_lm_loss = self.mlm_loss(
                    rearrange(prediction_scores, "... v -> (...) v"),
                    rearrange(labels, "... -> (...)"),
                )
            next_sentence_loss = self.nsp_loss(
                rearrange(seq_relationship_score, "... t -> (...) t"),
                rearrange(next_sentence_label, "... -> (...)"),
            )
            total_loss = masked_lm_loss.float() + next_sentence_loss.float()

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
        )


def remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items())

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^bert.encoder.layer.", "bert.encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^bert.embeddings.LayerNorm.", "bert.emb_ln.", key)
        key = re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"bert.encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^bert.encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^bert.encoder.layers.(\d+).output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(config, "last_layer_subset", False)
    for d in range(config.num_hidden_layers):
        Wq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"bert.encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.weight"] = torch.cat(
                [Wq, Wk, Wv], dim=0
            )
            state_dict[f"bert.encoder.layers.{d}.mixer.Wqkv.bias"] = torch.cat([bq, bk, bv], dim=0)
        else:
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.weight"] = Wq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.weight"] = torch.cat([Wk, Wv], dim=0)
            state_dict[f"bert.encoder.layers.{d}.mixer.Wq.bias"] = bq
            state_dict[f"bert.encoder.layers.{d}.mixer.Wkv.bias"] = torch.cat([bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub(
            r"^bert.encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"bert.encoder.layers.\1.mixer.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    state_dict = OrderedDict((key_mapping_decoder_bias(k), v) for k, v in state_dict.items())

    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        state_dict["bert.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
        )
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        state_dict["cls.predictions.decoder.weight"] = F.pad(
            decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0])
        )
        # If the vocab was padded, we want to set the decoder bias for those padded indices to be
        # strongly negative (i.e. the decoder shouldn't predict those indices).
        # TD [2022-05-09]: I don't think it affects the MLPerf training.
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        state_dict["cls.predictions.decoder.bias"] = F.pad(
            decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0
        )

    return state_dict


def inv_remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a flash_attn model to be Huggingface BERT compatible.

    This function is meant to be the inverse of remap_state_dict.
    """
    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["bert.embeddings.word_embeddings.weight"]
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        # unpad embeddings
        state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings[
            : config.orig_vocab_size, :
        ]
        state_dict["cls.predictions.decoder.weight"] = decoder_weight[: config.orig_vocab_size, :]
        state_dict["cls.predictions.decoder.bias"] = decoder_bias[: config.orig_vocab_size]

    for d in range(config.num_hidden_layers):
        last_layer_subset = getattr(config, "last_layer_subset", False)
        if not last_layer_subset or d != (config.num_hidden_layers - 1):
            Wqkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wqkv.weight")
            Wqkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wqkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = Wqkv_weights[
                : Wqkv_weights.shape[0] // 3, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = Wqkv_weights[
                Wqkv_weights.shape[0] // 3 : 2 * Wqkv_weights.shape[0] // 3, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = Wqkv_weights[
                2 * Wqkv_weights.shape[0] // 3 :, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = Wqkv_biases[
                : Wqkv_biases.shape[0] // 3
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = Wqkv_biases[
                Wqkv_biases.shape[0] // 3 : 2 * Wqkv_biases.shape[0] // 3
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = Wqkv_biases[
                2 * Wqkv_biases.shape[0] // 3 :
            ]
        else:
            Wq_weight = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wq.weight")
            Wkv_weights = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wkv.weight")
            Wq_bias = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wq.bias")
            Wkv_biases = state_dict.pop(f"bert.encoder.layers.{d}.mixer.Wkv.bias")
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.weight"] = Wq_weight
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.weight"] = Wkv_weights[
                : Wkv_weights.shape[0] // 2, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.weight"] = Wkv_weights[
                Wkv_weights.shape[0] // 2 :, :
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.query.bias"] = Wq_bias
            state_dict[f"bert.encoder.layers.{d}.attention.self.key.bias"] = Wkv_biases[
                : Wkv_biases.shape[0] // 2
            ]
            state_dict[f"bert.encoder.layers.{d}.attention.self.value.bias"] = Wkv_biases[
                Wkv_biases.shape[0] // 2 :
            ]

    def inv_key_mapping_ln(key):
        key = re.sub(r"bert.emb_ln.", "bert.embeddings.LayerNorm.", key)
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm1.(weight|bias)",
            r"bert.encoder.layers.\1.attention.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layers.(\d+).norm2.(weight|bias)",
            r"bert.encoder.layers.\1.output.LayerNorm.\2",
            key,
        )
        key = re.sub(
            r"cls.predictions.transform.layer_norm.(weight|bias)",
            r"cls.predictions.transform.LayerNorm.\1",
            key,
        )
        return key

    def inv_key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.weight$", "LayerNorm.gamma", key)
        key = re.sub(r"LayerNorm.bias$", "LayerNorm.beta", key)
        return key

    def inv_key_mapping_layers(key):
        return re.sub(r"bert.encoder.layers.", "bert.encoder.layer.", key)

    def inv_key_mapping_mlp(key):
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc1.(weight|bias)",
            r"bert.encoder.layer.\1.intermediate.dense.\2",
            key,
        )
        key = re.sub(
            r"bert.encoder.layer.(\d+).mlp.fc2.(weight|bias)",
            r"bert.encoder.layer.\1.output.dense.\2",
            key,
        )
        return key

    def inv_key_mapping_attn(key):
        return re.sub(
            r"bert.encoder.layer.(\d+).mixer.out_proj.(weight|bias)",
            r"bert.encoder.layer.\1.attention.output.dense.\2",
            key,
        )

    def inv_key_mapping_decoder_bias(key):
        return re.sub(r"cls.predictions.decoder.bias", "cls.predictions.bias", key)

    state_dict = OrderedDict((inv_key_mapping_ln(key), value) for key, value in state_dict.items())
    state_dict = OrderedDict(
        (inv_key_mapping_ln_gamma_beta(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_layers(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict((inv_key_mapping_mlp(key), value) for key, value in state_dict.items())
    state_dict = OrderedDict(
        (inv_key_mapping_attn(key), value) for key, value in state_dict.items()
    )
    state_dict = OrderedDict(
        (inv_key_mapping_decoder_bias(key), value) for key, value in state_dict.items()
    )

    return state_dict
