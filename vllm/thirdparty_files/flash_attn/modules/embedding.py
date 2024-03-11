# Copyright (c) 2022, Tri Dao.

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from flash_attn.utils.distributed import all_reduce, reduce_scatter


class GPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
            the project up to embed_dim
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs
            )
            self.project_in = nn.Linear(
                word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs
            )
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim, **factory_kwargs
            )

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        padding_idx=None,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        If type_vocab_size <= 0, there's no token type embeddings
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs
        )
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim, **factory_kwargs
            )
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_dim, **factory_kwargs)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        token_type_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        return embeddings


class VocabParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, *args, process_group=None, padding_idx=None, **kwargs):
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if num_embeddings % world_size != 0:
                raise ValueError(
                    f"num_embeddings ({num_embeddings}) must be divisible by "
                    f"world_size ({world_size})"
                )
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(num_embeddings // world_size, *args, padding_idx=padding_idx, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if self.process_group is None:
            return super().forward(input)
        else:
            rank = torch.distributed.get_rank(self.process_group)
            vocab_size = self.num_embeddings
            vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (input < vocab_start_index) | (input >= vocab_end_index)
            input = input - vocab_start_index
            input[input_ids_mask] = 0
            embeddings = super().forward(input)
            embeddings[input_ids_mask] = 0.0
            return embeddings


class ColumnParallelEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, *args, process_group=None, **kwargs):
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if embedding_dim % world_size != 0:
                raise ValueError(
                    f"embedding_dim ({embedding_dim}) must be divisible by "
                    f"world_size ({world_size})"
                )
        else:
            world_size = 1
        super().__init__(num_embeddings, embedding_dim // world_size, *args, **kwargs)


class ParallelGPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        process_group,
        padding_idx=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.word_embeddings = VocabParallelEmbedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx,
            process_group=process_group,
            **factory_kwargs,
        )
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = ColumnParallelEmbedding(
                max_position_embeddings, embed_dim, process_group=process_group, **factory_kwargs
            )

    def forward(self, input_ids, position_ids=None, combine_batch_seqlen_dim=False):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        world_size = torch.distributed.get_world_size(self.process_group)
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            if world_size <= 1:
                embeddings = embeddings + position_embeddings
            else:
                partition_dim = self.position_embeddings.embedding_dim
                rank = torch.distributed.get_rank(self.process_group)
                embeddings[
                    ..., rank * partition_dim : (rank + 1) * partition_dim
                ] += position_embeddings
        if combine_batch_seqlen_dim:
            embeddings = rearrange(embeddings, "b s d -> (b s) d")
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return embeddings if world_size <= 1 else reduce_fn(embeddings, self.process_group)
