# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.platforms import current_platform

from .utils import (dense_to_crow_col, get_head_sliding_step,
                    get_sparse_attn_mask)

IS_COMPUTE_8_OR_ABOVE = current_platform.has_device_capability(80)

if IS_COMPUTE_8_OR_ABOVE:
    from .blocksparse_attention_kernel import blocksparse_flash_attn_varlen_fwd


class LocalStridedBlockSparseAttn(torch.nn.Module):

    def __init__(
        self,
        n_heads,
        max_seqlen,
        local_blocks,
        vert_stride,
        block_size,
        device=None,
        dtype=None,
        homo_head=False,
        active_head_range=None,
        q_block_size=None,
        use_spda=None,
    ):
        super().__init__()
        if use_spda is None:
            use_spda = current_platform.is_rocm() or \
                        current_platform.is_cpu() or not \
                        IS_COMPUTE_8_OR_ABOVE
        device = device or (torch.cuda.current_device()
                            if current_platform.is_cuda_alike() else "cpu")
        device = torch.device(device)
        # NOTE: vllm CPU backend support BF16 instead of FP16.
        dtype = dtype or (torch.bfloat16 if IS_COMPUTE_8_OR_ABOVE
                          or device.type == "cpu" else torch.half)

        self.n_heads = n_heads
        self.max_seqlen = max_seqlen
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        self.use_spda = use_spda
        self.dtype = dtype
        self.device = device
        self.block_size = block_size
        self.q_block_size = q_block_size
        self.homo_head = homo_head
        self.active_head_range = active_head_range
        self.head_sliding_step = get_head_sliding_step(n_heads, vert_stride,
                                                       homo_head)

        sparse_layout, sparse_pattern, self.dense_attn_mask = (
            self.get_attn_pattern(dtype, device))

        if q_block_size is not None and q_block_size != block_size:
            if q_block_size > block_size:
                assert q_block_size % block_size == 0
                blocks_to_merge = q_block_size // block_size
                shape = sparse_pattern.shape
                sparse_pattern = sparse_pattern.view(shape[0], -1,
                                                     blocks_to_merge,
                                                     shape[-1])
                sparse_pattern = sparse_pattern.sum(2)
                sparse_layout = dense_to_crow_col(sparse_pattern)
            else:
                raise ValueError(
                    "Does not support smaller q_block_size. It will be slower."
                )

        self.sparse_layout = sparse_layout

    def get_attn_pattern(self, dtype, device):
        sparse_layout, sparse_pattern, dense_attn_mask = get_sparse_attn_mask(
            self.n_heads,
            self.max_seqlen,
            self.max_seqlen,
            dtype,
            device,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
            vert_stride=self.vert_stride,
            homo_head=self.homo_head,
            return_dense=self.use_spda,
            dense_mask_type="bias",
        )
        if (not self.homo_head) and (self.active_head_range is not None):
            assert isinstance(self.active_head_range, tuple)
            assert (len(self.active_head_range) == 2)
            h_start, h_end = self.active_head_range
            sparse_layout = tuple(x[h_start:h_end] for x in sparse_layout)
            if self.use_spda:
                dense_attn_mask = dense_attn_mask[h_start:h_end]
        return sparse_layout, sparse_pattern, dense_attn_mask

    def varlen_attn(self,
                    q,
                    k,
                    v,
                    cu_seqlens_k,
                    cu_seqlens_q=None,
                    sm_scale=None):
        """
        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
        Support grouped attention, with `q[:, i*r:(i*r + r)]`
        is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,),
        indicating segment of samples,
        e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
        Default None: same as cu_seqlens_k for prefilling or
        [0, 1, .., batch_size] for decoding.
        The only case you need to specify is when q is a mix of
        prefilling and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        """
        assert (
            IS_COMPUTE_8_OR_ABOVE
        ), "Requires compute capability of 8 or above (Ampere or newer) to use \
            Triton kernel."

        sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))

        return blocksparse_flash_attn_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_k,
            cu_seqlens_q,
            sm_scale,
            self.sparse_layout,
            block_size=self.block_size,
            q_block_size=self.q_block_size,
            max_seqlen=self.max_seqlen,
        )

    @staticmethod
    def transpose_and_pad(x, cu_seqlens, maxlen, head_repeats=1):
        """
        :param x: (total_tokens, n_heads, head_size)
        :return: (batch, n_heads, length, head_size)
        """
        x_padded = x.new_empty(
            len(cu_seqlens) - 1, x.size(1), head_repeats, maxlen, x.size(2))
        cu_seqlens = cu_seqlens.cpu()
        for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            x_padded[i, :, :, :e - s].copy_(x[s:e].transpose(0,
                                                             1).unsqueeze(1))
        return x_padded.flatten(1, 2)

    @staticmethod
    def transpose_and_unpad(x_padded, cu_seqlens):
        """
        :param x_padded: (batch, n_heads, length, head_size)
        :return: (total_tokens, n_heads, head_size)
        """
        cu_seqlens = cu_seqlens.cpu()
        total_n_tokens = cu_seqlens[-1]
        x = x_padded.new_empty(total_n_tokens, x_padded.size(1),
                               x_padded.size(3))
        for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            x[s:e].copy_(x_padded[i, :, :e - s].transpose(0, 1))
        return x

    def spda(self, q, k, v, cu_seqlens_k, cu_seqlens_q=None, sm_scale=None):
        """For CPU, V100 or other older GPUs.
        NOTE: torch SPDA supports nested tensor,
        but seems extremely slow. Choose to pad instead.
        """
        assert (cu_seqlens_q is None or
                (cu_seqlens_q
                 == cu_seqlens_k).all()), "Can only handle prompt with SPDA."
        assert q.size(0) == k.size(0), "can only handle prompt with SPDA."

        assert q.size(1) % k.size(1) == 0
        q_k_ratio = q.size(1) // k.size(1)
        sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))
        cu_seqlens = cu_seqlens_k.cpu()
        maxlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        if (self.dense_attn_mask.dtype != q.dtype
                or self.dense_attn_mask.device != q.device):
            _, _, self.dense_attn_mask = self.get_attn_pattern(
                q.dtype, q.device)
        attn_mask = self.dense_attn_mask[None, :, :maxlen, :maxlen]

        q2 = self.transpose_and_pad(q, cu_seqlens, maxlen, 1)
        k2, v2 = (self.transpose_and_pad(x, cu_seqlens, maxlen, q_k_ratio)
                  for x in [k, v])
        spda_output = torch.nn.functional.scaled_dot_product_attention(
            q2, k2, v2, attn_mask=attn_mask, scale=sm_scale)
        return self.transpose_and_unpad(spda_output, cu_seqlens)

    def forward(self, q, k, v, cu_seqlens_k, cu_seqlens_q=None, sm_scale=None):
        """Dispatch to `varlen_attn` (Ampere or newer) or
        `self.spda`(cpu, Volta, Turing or older)based on
        the type of device used and cuda compute capability.

        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
                Support grouped attention, with `q[:, i*r:(i*r + r)]`
                is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,), indicating segment of samples,
                    e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
                    Default None: same as cu_seqlens_k for prefilling or
                    [0, 1, .., batch_size] for decoding.
                    The only case you need to specify
                    is when q is a mix of prefilling
                    and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        """
        assert k.dim() == 3
        if self.use_spda:
            return self.spda(
                q,
                k,
                v,
                cu_seqlens_k,
                cu_seqlens_q=cu_seqlens_q,
                sm_scale=sm_scale,
            )
        return self.varlen_attn(q,
                                k,
                                v,
                                cu_seqlens_k,
                                cu_seqlens_q=cu_seqlens_q,
                                sm_scale=sm_scale)
