import torch
import math
from functools import lru_cache
from .utils import get_sparse_attn_mask, dense_to_crow_col

IS_COMPUTE_8_OR_ABOVE = (torch.cuda.is_available()
                         and torch.cuda.get_device_capability()[0] >= 8)

if IS_COMPUTE_8_OR_ABOVE:
    from .kernels import blocksparse_flash_attn_varlen_fwd, blocksparse_flash_attn_varlen_fwd_with_blocktable


class LocalStridedBlockSparseAttn(torch.nn.Module):

    def __init__(self,
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
                 use_spda=None):
        super().__init__()
        if use_spda is None:
            use_spda = not (torch.cuda.is_available()
                            and torch.cuda.get_device_capability()[0] >= 8)
        device = device or (torch.cuda.current_device()
                            if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        # NOTE: vllm CPU backend support BF16 instead of FP16.
        dtype = dtype or (torch.bfloat16 if IS_COMPUTE_8_OR_ABOVE
                          or device.type == 'cpu' else torch.half)

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

        sparse_layout, sparse_pattern, self.dense_attn_mask = self.get_attn_pattern(
            dtype, device)

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
                    'Does not support smaller q_block_size. It will be slower.'
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
            dense_mask_type='bias')
        if (not self.homo_head) and (self.active_head_range is not None):
            assert isinstance(self.active_head_range, tuple)
            assert len(
                self.active_head_range
            ) == 2, '"active_head_range" should be a tuple of start/end index of the heads.'
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
        '''
        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
                Support grouped attention, with `q[:, i*r:(i*r + r)]`
                is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,), indicating segment of samples, e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
                    Default None: same as cu_seqlens_k for prefilling or
                    [0, 1, .., batch_size] for decoding.
                    The only case you neeed to specify is when q is a mix of prefilling and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        '''
        assert IS_COMPUTE_8_OR_ABOVE, 'Requires compute capability of 8 or above (Ampere or newer) to use Triton kernel.'
        sm_scale = sm_scale or 1. / math.sqrt(q.size(-1))

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
            max_seqlen=self.max_seqlen)

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
        '''For CPU, V100 or other older GPUs.
        NOTE: torch SPDA supports nested tensor, but seems extremely slow. Choose to pad instead.
        '''
        assert cu_seqlens_q is None or (
            cu_seqlens_q
            == cu_seqlens_k).all(), "Can only handle prompt with SPDA."
        assert q.size(0) == k.size(0), "can only handle prompt with SPDA."

        assert q.size(1) % k.size(1) == 0
        q_k_ratio = q.size(1) // k.size(1)
        sm_scale = sm_scale or 1. / math.sqrt(q.size(-1))
        cu_seqlens = cu_seqlens_k.cpu()
        maxlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        if self.dense_attn_mask.dtype != q.dtype or self.dense_attn_mask.device != q.device:
            _, _, self.dense_attn_mask = self.get_attn_pattern(
                q.dtype, q.device)
        attn_mask = self.dense_attn_mask[None, :, :maxlen, :maxlen]

        q2 = self.transpose_and_pad(q, cu_seqlens, maxlen, 1)
        k2, v2 = [
            self.transpose_and_pad(x, cu_seqlens, maxlen, q_k_ratio)
            for x in [k, v]
        ]
        spda_output = torch.nn.functional.scaled_dot_product_attention(
            q2, k2, v2, attn_mask=attn_mask, scale=sm_scale)
        return self.transpose_and_unpad(spda_output, cu_seqlens)

    def forward(self, q, k, v, cu_seqlens_k, cu_seqlens_q=None, sm_scale=None):
        '''Dispatch to `varlen_attn` (Ampere or newer) or `self.spda`(cpu, Volta, Turing or older)
        based on the type of device used and cuda compute capability.

        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
                Support grouped attention, with `q[:, i*r:(i*r + r)]`
                is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        cu_seqlens_k: shape=(batch_size + 1,), indicating segment of samples,
                    e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
        cu_seqlens_q: shape=(batch_size + 1, ).
                    Default None: same as cu_seqlens_k for prefilling or
                    [0, 1, .., batch_size] for decoding.
                    The only case you neeed to specify is when q is a mix of prefilling and decoding.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        '''
        assert k.dim() == 3
        if self.use_spda:
            return self.spda(q,
                             k,
                             v,
                             cu_seqlens_k,
                             cu_seqlens_q=cu_seqlens_q,
                             sm_scale=sm_scale)
        return self.varlen_attn(q,
                                k,
                                v,
                                cu_seqlens_k,
                                cu_seqlens_q=cu_seqlens_q,
                                sm_scale=sm_scale)


class LocalStridedBlockSparsePagedAttn(torch.nn.Module):

    def __init__(self,
                 n_heads,
                 max_seqlen,
                 local_blocks,
                 vert_stride,
                 block_size,
                 device=None,
                 dtype=torch.bfloat16,
                 homo_head=False,
                 active_head_range=None,
                 vllm_block_size=None,
                 mode='split'):
        super().__init__()
        device = device or torch.cuda.current_device()
        self.max_seqlen = max_seqlen
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        sparse_layout, sparse_pattern, _ = get_sparse_attn_mask(
            n_heads,
            max_seqlen,
            max_seqlen,
            dtype,
            device,
            block_size=block_size,
            local_blocks=local_blocks,
            vert_stride=vert_stride,
            homo_head=homo_head,
            return_dense=False)
        self.mode = mode
        if mode in ('split', 'remote-only'):
            sparse_layout, sparse_pattern = self.get_remote_sparse_layout(
                n_heads,
                max_seqlen,
                dtype,
                device,
                block_size,
                local_blocks,
                vert_stride,
                homo_head=homo_head,
                return_dense=False)

        if (not homo_head) and (active_head_range is not None):
            assert isinstance(active_head_range, tuple)
            assert len(
                active_head_range
            ) == 2, '"active_head_range" should be a tuple of start/end index of the heads.'
            h_start, h_end = active_head_range
            sparse_layout = tuple(x[h_start:h_end] for x in sparse_layout)
            sparse_pattern = sparse_pattern[h_start:h_end]

        self.sparse_layout = sparse_layout
        self.sparse_pattern = sparse_pattern

        self.vllm_block_size = None
        if vllm_block_size:
            self.set_vllm_block_size(vllm_block_size)

    def set_vllm_block_size(self, vllm_block_size):
        if self.vllm_block_size is not None:
            raise ValueError('vllm_block_size has been set')

        self.vllm_block_size = vllm_block_size
        sparse_block_size = self.block_size
        kernel_block_size = vllm_block_size

        assert sparse_block_size % kernel_block_size == 0
        if sparse_block_size // kernel_block_size > 1:
            _mul = sparse_block_size // kernel_block_size
            sparse_pattern = torch.kron(
                self.sparse_pattern, self.sparse_pattern.new_ones(_mul, _mul))
            num_sparse_blocks = sparse_pattern.size(-1)
            block_causal_mask = torch.arange(
                0, num_sparse_blocks)[:, None] >= torch.arange(
                    0, num_sparse_blocks)[None]
            sparse_pattern *= block_causal_mask.type_as(sparse_pattern)
            sparse_layout = dense_to_crow_col(sparse_pattern)
            self.sparse_layout = sparse_layout
            self.sparse_pattern = self.sparse_pattern

    @lru_cache
    def get_remote_sparse_layout(self,
                                 n_heads,
                                 max_seqlen,
                                 dtype,
                                 device,
                                 block_size,
                                 local_blocks,
                                 vert_stride,
                                 homo_head=False,
                                 return_dense=False):
        _, sparse_pattern, _ = get_sparse_attn_mask(n_heads,
                                                    max_seqlen,
                                                    max_seqlen,
                                                    dtype,
                                                    device,
                                                    block_size=block_size,
                                                    local_blocks=local_blocks,
                                                    vert_stride=vert_stride,
                                                    homo_head=homo_head,
                                                    return_dense=False)

        _, sparse_pattern_local, _ = get_sparse_attn_mask(
            n_heads,
            max_seqlen,
            max_seqlen,
            dtype,
            device,
            block_size=block_size,
            local_blocks=local_blocks,
            vert_stride=max_seqlen + 1,
            homo_head=homo_head,
            return_dense=return_dense)
        sparse_pattern_strides = sparse_pattern - sparse_pattern_local

        sparse_layout_strides = dense_to_crow_col(sparse_pattern_strides)
        return sparse_layout_strides, sparse_pattern_strides

    def forward(self,
                q,
                k,
                v,
                block_tables,
                context_lens,
                sm_scale=None,
                kv_scale=1.0):
        '''
        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
                Support grouped attention, with `q[:, i*r:(i*r + r)]`
                is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        '''
        if self.sparse_layout[0].size(0) != 1:
            assert q.size(1) == self.sparse_layout[0].size(0)

        sm_scale = sm_scale or 1. / math.sqrt(q.size(-1))

        if self.vllm_block_size is None:
            self.set_vllm_block_size(v.size(-1))

        # TODO: auto extend length to next_power_of_2
        assert block_tables.size(1) * self.vllm_block_size <= self.max_seqlen

        return blocksparse_flash_attn_varlen_fwd_with_blocktable(
            q,
            k,
            v,
            block_tables,
            context_lens,
            sm_scale,
            self.sparse_layout,
            sparse_block_size=self.block_size,
            vllm_block_size=self.vllm_block_size,
            num_local_blocks=self.local_blocks,
            mode=self.mode,
            max_seqlen=self.max_seqlen,
            kv_scale=kv_scale)
