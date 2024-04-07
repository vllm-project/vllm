"""Attention layer with xFormers and PagedAttention."""
import importlib
from typing import List, Optional

import torch
import torch.distributed
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias,
                                         LowerTriangularFromBottomRightMask)

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.paged_attn import (
    PagedAttentionImpl)
from vllm.utils import is_hip


class XFormersBackend:

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = PagedAttentionImpl.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.use_ref_attention = _check_use_ref_attention()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        status: int,
        cache_fuse_metadata: dict,
        cache_load_metadata: dict,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape        
        
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        
        
        # Do checking here
        if status in [1]:
            # Get cached KV

            # import pdb
            # pdb.set_trace()
            
            value_old = torch.empty_like(value)
            key_old = torch.empty_like(key)
            # print("We are saving kv for shape: ", value_old.shape)
            # torch.save(value_old, f"/local/hanchen/kv_temp/{str(cache_load_metadata['layer'])}_0")
            # torch.save(key_old, f"/local/hanchen/kv_temp/{str(cache_load_metadata['layer'])}_1")

            #FIXME Hanchen need to add cuda device. also change to directly load into GPU memory

            #key_old = cache_load_metadata['loader'].fetch_kv_layer(cache_load_metadata['hash'],
            #                                                        cache_load_metadata['layer'], True, 'cuda:0')
            #value_old =  cache_load_metadata['loader'].fetch_kv_layer(cache_load_metadata['hash'],
            #                                                        cache_load_metadata['layer'], False, 'cuda:0')
            
            # if (value_old is None or key_old is None):
            #     exit("KV failure")

            # pdb.set_trace()

            #FIXME(Jiayi): Optimize this kernel to only load value_old or even lesser stuff
            PagedAttentionImpl.load_and_reshape(key_old, value_old, key_cache,
                                                 value_cache, cache_fuse_metadata)
            # Get deviations
            topk_num = int(value.shape[0]*cache_fuse_metadata["recomp_ratio"])

            temp_diff = torch.sum((value[:,:,:]-value_old[:,:,:])**2,dim=[1,2])
            top_indices = torch.topk(temp_diff, k=topk_num).indices

            #top_indices = top_indices#.cpu().numpy().tolist()
            xxx = [torch.zeros_like(top_indices), torch.zeros_like(top_indices)]
            #torch.distributed.barrier()
            torch.distributed.all_gather(xxx, top_indices)
            #torch.distributed.barrier()
            #print("xxx[0]:", xxx[0])
            #print("xxx[1]:", xxx[1])
            top_indices = torch_intersect1d(xxx[0],xxx[1])[:topk_num]
            top_indices = top_indices.sort().values #Jiayi: only need this if use one node
            #print(top_indices)
            
            # Add last token idx if not in topk
            if seq_len-1 not in top_indices:
                top_indices = torch.cat([top_indices,torch.tensor([seq_len-1], device=top_indices.device)])
            
            # Construct our slot mapping
            our_slot_mapping = input_metadata.slot_mapping[:,top_indices]
            
            #FIXME(Jiayi): this can be faster
            #our_slot_mapping_for_check = input_metadata.slot_mapping.clone()
            #our_slot_mapping_for_check[:,list(set([i for i in range(seq_len)])-set(top_indices))] = -1
            
            cache_fuse_metadata["our_slot_mapping"] = our_slot_mapping
            #cache_fuse_metadata["our_slot_mapping_for_check"] = our_slot_mapping_for_check
            
            # reduce query shape
            query = query[top_indices]
            
            # Assign imp_token_indices
            cache_fuse_metadata["imp_token_indices"] = top_indices
            
            # Construct mask (attn bias)
            #attn_bias = _make_partial_bias(cache_fuse_metadata, query.device, self.num_heads)
            attn_bias = _make_partial_bias_70b(cache_fuse_metadata, query.device, self.num_kv_heads, self.num_queries_per_kv)
            #attn_bias = _fetch_partial_pre_mask(cache_fuse_metadata)
            cache_fuse_metadata["attn_bias"] = attn_bias
            
            #This is to save memory
            input_metadata.attn_bias=None
        #if cache_fuse_metadata["check"]:
        #    print(len(cache_fuse_metadata["imp_token_indices"]))
        #if torch.distributed.get_rank()==1 and status in [1]:
        #    import pdb
        #    pdb.set_trace()
        #torch.distributed.barrier()
        
        '''
        if seq_len>4000 or cache_fuse_metadata["org_seq_len"]>4000:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        '''
          
        #import pdb
        #pdb.set_trace()
        if status in [0] and cache_fuse_metadata["original_slot_mapping"]==None and cache_fuse_metadata['check']:
            #import pdb
            #pdb.set_trace()
            cache_fuse_metadata["original_slot_mapping"] = input_metadata.slot_mapping
            cache_fuse_metadata["key_shape"] = key.shape
            cache_fuse_metadata["value_shape"] = value.shape
            cache_fuse_metadata["kv_cache_string_dtype"] = "auto"
            cache_fuse_metadata["kv_cache_dtype"] = value.dtype
            #FIXME(Jiayi): need a hack in the inference test script to do prefill for chunks
        
        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            if cache_fuse_metadata['check'] and status in [1,2]:
                if status in [2]:
                    #key and value are 15% recomputed, key_cache full original, update 15% of key cache.
                    #Key value 100% original
                    PagedAttentionImpl.reshape_and_cache_ours(key, value, key_cache,
                                                    value_cache, cache_fuse_metadata)
                elif status in [1]:
                    #Jiayi: use _for_check for partial update
                    #input_metadata.slot_mapping = cache_fuse_metadata["ouriginal"]
                    PagedAttentionImpl.reshape_and_cache(key, value, key_cache,
                                                    value_cache, input_metadata)
                    #input_metadata.slot_mapping = cache_fuse_metadata["original_slot_mapping"]
            else:
                PagedAttentionImpl.reshape_and_cache(key, value, key_cache,
                                                value_cache, input_metadata)
        '''
        if seq_len>4000 or cache_fuse_metadata["org_seq_len"]>4000:
            end.record()
            torch.cuda.synchronize()
            temp_time = start.elapsed_time(end)
            print(f"Cache store time:{temp_time}")
        '''    
        
        '''
        if seq_len>4000 or cache_fuse_metadata["org_seq_len"]>4000:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        '''
        # FIXME(Jiayi): can we do kernel fusion here?
        if status in [2]: #load memory if `after_check`
            key = torch.empty(cache_fuse_metadata["key_shape"], 
                               dtype=key.dtype, 
                               device=key.device)
            # torch.save(key, f"/local/hanchen/kv_temp/{str(cache_load_metadata['layer'])}_1")
            value = torch.empty(cache_fuse_metadata["value_shape"],
                                dtype=value.dtype, 
                                device=value.device)
            # torch.save(value, f"/local/hanchen/kv_temp/{str(cache_load_metadata['layer'])}_0")
            
            #key = cache_load_metadata['loader'].fetch_kv_layer(cache_load_metadata['hash'],
            #                                                    cache_load_metadata['layer'], True, 'cuda:0')
            #value =  cache_load_metadata['loader'].fetch_kv_layer(cache_load_metadata['hash'],
            #                                                        cache_load_metadata['layer'], False, 'cuda:0')
            # assert(value_old.shape == key.shape)
            #if (value is None or key is None):
            #    exit("KV failure")
            #import pdb
            #pdb.set_trace()
            PagedAttentionImpl.load_and_reshape(key, value, key_cache,
                                                 value_cache, cache_fuse_metadata)
            
        '''
        if seq_len>4000 or cache_fuse_metadata["org_seq_len"]>4000:
            end.record()
            torch.cuda.synchronize()
            temp_time = start.elapsed_time(end)
            print(f"Cache load time:{temp_time}")
        if seq_len>4000 or cache_fuse_metadata["org_seq_len"]>4000:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        '''
        if input_metadata.is_prompt:
            # Prompt run.
            if (key_cache is None or value_cache is None
                    or input_metadata.block_tables.numel() == 0):
                
                # normal attention
                if self.num_kv_heads != self.num_heads:
                    # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                    # project the key and value tensors to the desired number of
                    # heads.
                    # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                    query = query.view(query.shape[0], self.num_kv_heads,
                                       self.num_queries_per_kv,
                                       query.shape[-1])
                    key = key[:, :,
                              None, :].expand(key.shape[0], self.num_kv_heads,
                                              self.num_queries_per_kv,
                                              key.shape[-1])
                    value = value[:, :,
                                  None, :].expand(value.shape[0],
                                                  self.num_kv_heads,
                                                  self.num_queries_per_kv,
                                                  value.shape[-1])

                # Set attention bias if not provided. This typically happens at
                # the very attention layer of every iteration.
                # FIXME(woosuk): This is a hack.
                if input_metadata.attn_bias is None:
                    if self.alibi_slopes is None:
                        attn_bias = BlockDiagonalCausalMask.from_seqlens(
                            [seq_len] * batch_size)
                        #import pdb
                        #pdb.set_trace()
                        #HACK(Jiayi): sdpa
                        #attn_bias = _make_pre_mask(seq_len, query.dtype, query.device, self.num_heads)
                        if self.sliding_window is not None:
                            attn_bias = attn_bias.make_local_attention(
                                self.sliding_window)
                        #HACK(Jiayi): force
                        #input_metadata.attn_bias = _fetch_maetrailized_mask(query.shape[0],key.shape[0], self.num_heads, attn_bias,query.device,query.dtype)
                        input_metadata.attn_bias = _fetch_maetrailized_mask_70b(query.shape[0], self.num_kv_heads, self.num_queries_per_kv, query.device,query.dtype)
                    else:
                        input_metadata.attn_bias = _make_alibi_bias(
                            self.alibi_slopes, self.num_kv_heads, batch_size,
                            seq_len, query.dtype)

                if self.use_ref_attention:
                    output = _ref_masked_attention(
                        query,
                        key,
                        value,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_size,
                        self.scale,
                    )
                    # Using view got RuntimeError: view size is not compatible
                    # with input tensor's size and stride (at least one
                    # dimension spans across two contiguous subspaces).
                    # Use reshape instead.
                    return output.reshape(batch_size, seq_len, hidden_size)

                # TODO(woosuk): Too many view operations. Let's try to reduce
                # them in the future for code readability.
                if self.alibi_slopes is None:
                    query = query.unsqueeze(0)
                    key = key.unsqueeze(0)
                    value = value.unsqueeze(0)
                else:
                    query = query.unflatten(0, (batch_size, seq_len))
                    key = key.unflatten(0, (batch_size, seq_len))
                    value = value.unflatten(0, (batch_size, seq_len))
                                
                # Bias to apply to the attention matrix - defaults to no masking. 
                # For common biases implemented efficiently in xFormers, see xformers.ops.fmha.attn_bias.AttentionBias. 
                # This can also be a torch.Tensor for an arbitrary mask (slower)
                
                #FIXME(Jiayi): Please do not use materialized mask (See WeChat screenshot)
                # Assign dynamic attention mask
                if status in [1,2]:
                    out = xops.memory_efficient_attention_forward(
                        query,
                        key,
                        value,
                        attn_bias=cache_fuse_metadata["attn_bias"],
                        p=0.0,
                        scale=self.scale,
                        op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
                        (is_hip()) else None,
                    )
                else:
                    #print(input_metadata.attn_bias.shape)
                    out = xops.memory_efficient_attention_forward(
                        query,
                        key,
                        value,
                        attn_bias=input_metadata.attn_bias,
                        p=0.0,
                        scale=self.scale,
                        op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
                        (is_hip()) else None,
                    )
                


                '''
                #sdpa
                out = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=input_metadata.attn_bias,
                    dropout_p = 0.0,
                )
                '''
                
                #xops.memory_efficient_attention_forward(query,key,value,attn_bias=input_metadata.attn_bias,p=0.0,scale=self.scale,op=None,)
                #if cache_fuse_metadata["org_seq_len"]==11:
                #    import pdb
                #    pdb.set_trace()
                
                
                output = out.view_as(query)
                
                #Jiayi: this is for the `return output.view(batch_size, seq_len, hidden_size)` statement below
                if status in [1]:
                    seq_len = len(top_indices)

            else:
                # prefix-enabled attention
                output = PagedAttentionImpl.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    input_metadata,
                    self.alibi_slopes,
                )
        else:
            # Decoding run.
            output = PagedAttentionImpl.forward_decode(
                query,
                key_cache,
                value_cache,
                input_metadata,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )
        '''
        if seq_len>4000 or cache_fuse_metadata["org_seq_len"]>4000:
            end.record()
            torch.cuda.synchronize()
            temp_time = start.elapsed_time(end)
            print(f"Attn comp time:{temp_time}")
        '''
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias

def _fetch_maetrailized_mask(q_len,k_len,num_head,mask,device,dtype):
    seq_len = q_len
    padded_len = (seq_len + 7) // 8 * 8
    attn_mask = torch.triu(torch.ones(seq_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 1, seq_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask.expand(-1,num_head,-1,-1)
    
    attn_mask_padded = torch.empty(
        1,
        num_head,
        seq_len,
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[:, :, :, :seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded

def _fetch_maetrailized_mask_70b(q_len,num_kv_heads,num_queries_per_kv,device,dtype):
    seq_len = q_len
    padded_len = (seq_len + 7) // 8 * 8
    attn_mask = torch.triu(torch.ones(seq_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 1, 1, seq_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask.expand(1,num_kv_heads, num_queries_per_kv,-1,-1)
    
    attn_mask_padded = torch.empty(
        1,
        num_kv_heads,
        num_queries_per_kv,
        seq_len,
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[:, :, :, :, :seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded



def _make_partial_bias(cache_fuse_metadata, 
                       device,
                       num_heads):
    seq_len = cache_fuse_metadata['org_seq_len']
    padded_len = (seq_len + 7) // 8 * 8
    dtype = cache_fuse_metadata['kv_cache_dtype']
    imp_indices = cache_fuse_metadata['imp_token_indices']
    attn_mask = torch.triu(torch.ones(padded_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 1, padded_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask[:,:,imp_indices]
    attn_mask = attn_mask.expand(-1,num_heads,-1,-1)
    
    attn_mask_padded = torch.empty(
        1,
        num_heads,
        len(imp_indices),
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[:, :, :, :seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded

def _make_partial_bias_70b(cache_fuse_metadata, 
                       device,
                       num_kv_heads,
                       num_queries_per_kv):
    seq_len = cache_fuse_metadata['org_seq_len']
    padded_len = (seq_len + 7) // 8 * 8
    dtype = cache_fuse_metadata['kv_cache_dtype']
    imp_indices = cache_fuse_metadata['imp_token_indices']
    attn_mask = torch.triu(torch.ones(padded_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 1, 1, padded_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask[:,:,:,imp_indices]
    attn_mask = attn_mask.expand(1,num_kv_heads,num_queries_per_kv,-1,-1)
    
    attn_mask_padded = torch.empty(
        1,
        num_kv_heads,
        num_queries_per_kv,
        len(imp_indices),
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[:, :, :, :,:seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded

def _check_use_ref_attention() -> bool:
    if not is_hip():
        return False
    # For ROCm, check whether flash attention is installed or not.
    # if not, use_ref_attention needs to be True
    return importlib.util.find_spec("flash_attn") is None


def _ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
) -> torch.Tensor:
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    seq_len, _, _ = query.shape
    attn_mask = torch.triu(torch.ones(seq_len,
                                      seq_len,
                                      dtype=query.dtype,
                                      device=query.device),
                           diagonal=1)
    #attn_mask = torch.triu(torch.ones(seq_len,seq_len,dtype=query.dtype,device=query.device),diagonal=1)
    attn_mask = attn_mask * torch.finfo(query.dtype).min

    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out

def _ref_masked_attention_2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    attn_mask,
) -> torch.Tensor:
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)


    #attn_mask = torch.triu(torch.ones(seq_len,seq_len,dtype=query.dtype,device=query.device),diagonal=1)
    attn_mask = attn_mask * torch.finfo(query.dtype).min

    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out

def torch_intersect1d(t1: torch.Tensor, t2: torch.Tensor):
    # NOTE: requires t1, t2 to be unique 1D Tensor in advance.
    # Method: based on unique's count
    num_t1 = t1.numel()
    _, inv, cnt = torch.unique(torch.cat([t1,t2]), return_counts=True, return_inverse=True)
    
    cnt_12 = cnt[inv]
    cnt_t2 = cnt_12[num_t1:]

    inds_t2_exclusive = (cnt_t2 == 1).nonzero()[..., 0]

    t2_exclusive = t2[inds_t2_exclusive]
    return torch.cat([t1,t2_exclusive])