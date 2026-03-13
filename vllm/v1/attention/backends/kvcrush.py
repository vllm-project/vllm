"""
KVCrush cache eviction policy
"""

import torch, math
from vllm.logger import init_logger
logger = init_logger(__name__)

class H2OKVCrushCluster:
    def __init__(self, window_size = 16, recent_size = 128, max_capacity_prompt = 256 + 64, kvcrush_ratio=0.25, start_size=32, anchor_mode="alternate", block_size=16):
        self.window_size = window_size
        self.recent_size = recent_size
        self.start_size = start_size
        self.max_capacity_prompt = max_capacity_prompt
        self.anchor_mode = anchor_mode
        self.kvcrush_ratio = kvcrush_ratio
        logger.info("H2O max_capacity_prompt %s", self.max_capacity_prompt)
        self._cached_causal_mask = None
        self._cached_causal_mask_size = 0

        assert self.recent_size % block_size == 0, f"recent_size must be multiple of {block_size}"
        assert self.start_size % block_size == 0, f"start_size must be multiple of {block_size}"
        assert self.max_capacity_prompt % block_size == 0, f"max_capacity_prompt must be multiple of {block_size}"


    def reset(self, window_size = 16, recent_size = 128, max_capacity_prompt = 256 + 64, start_size=0):
        self.window_size = window_size
        self.recent_size = recent_size
        self.start_size = start_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.recent_size - self.start_size > 0

    def _convert_group_indices(self, group_indices, seq_len, page_size, num_kv_heads):
        """Convert group indices to token indices, following cache_processor.py pattern"""
        num_groups = group_indices.shape[-1]
        # Create indices tensor for a single group
        idx = torch.arange(page_size, device=group_indices.device)
        idx = idx.repeat(num_kv_heads * num_groups).view(num_kv_heads, num_groups, page_size)
        expanded_mat = group_indices.unsqueeze(3)
        expanded_mat = expanded_mat.repeat(1, 1, 1, page_size)

        # Compute indices for each element in the input matrix
        indices = expanded_mat * page_size + idx
        indices = indices.view(1, num_kv_heads, num_groups * page_size)

        # Drop extra indices from the last group that are empty
        last_group_occupied = seq_len % page_size
        if last_group_occupied:
            padded = page_size - last_group_occupied
            indices = indices[:, :, :-padded]

        return indices

    def update_kv(self, key_states, query_states, value_states, page_size):
        # print("KVCrush update_kv called")
        # kvcrush - parameters
        seqlen = key_states.shape[-2]
        heads = key_states.shape[1]
        assert(page_size <= self.recent_size)

        # print("key_states.shape:", key_states.shape)
        # print("query_states.shape:", query_states.shape)
        # print("value_states.shape:", value_states.shape)

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        # Note: During decode, key_states has all cached tokens, query_states has only new tokens
        bsz, num_heads, q_len, head_dim = query_states.shape
        k_len = key_states.shape[-2]

        # Save original key/value states (in GQA format)
        key_states_original = key_states
        value_states_original = value_states

        # Expand temporarily for attention computation only
        num_kv_heads = key_states.shape[1]
        num_query_heads = query_states.shape[1]
        num_query_heads_per_kv = num_query_heads // num_kv_heads

        if num_query_heads_per_kv > 1:
            key_states_expanded = key_states.unsqueeze(2).expand(
                -1, -1, num_query_heads_per_kv, -1, -1
            ).reshape(bsz, num_query_heads, k_len, head_dim)
            value_states_expanded = value_states.unsqueeze(2).expand(
                -1, -1, num_query_heads_per_kv, -1, -1
            ).reshape(bsz, num_query_heads, k_len, head_dim)
        else:
            key_states_expanded = key_states
            value_states_expanded = value_states

        # Skip compression if sequences are empty or too short
        if k_len == 0 or q_len == 0 or k_len < self.max_capacity_prompt:
            return key_states_original, value_states_original
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)
            # Use cached causal mask to avoid re-creating every call
            if self._cached_causal_mask is None or self._cached_causal_mask_size != self.window_size:
                mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                self._cached_causal_mask = mask
                self._cached_causal_mask_size = self.window_size
            attention_mask = self._cached_causal_mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights.sum(dim = -2)

            # Aggregate attention scores across query heads that share the same KV head
            if num_query_heads_per_kv > 1:
                # Reshape to [bsz, num_kv_heads, num_query_heads_per_kv, seq_len]
                attn_cache_reshaped = attn_weights_sum.reshape(bsz, num_kv_heads, num_query_heads_per_kv, -1)
                # Average across query heads that share the same KV head
                attn_cache = attn_cache_reshaped.mean(dim=2)  # [bsz, num_kv_heads, seq_len]
            else:
                attn_cache = attn_weights_sum
                #recalculate budget
                # factor_ = 0.000001
                # mean_ = torch.mean(attn_weights)
                # while True:
                #     threshold_ = factor_ * mean_
                #     small_values_factor = torch.sum(attn_weights < threshold_).item() / attn_weights.numel()
                #     if (small_values_factor >= .25) or (factor_ > 0.1):
                #         break
                #     factor_ = factor_ * 10
                # budget_clus = int(self.max_capacity_prompt * min(self.kvcrush_ratio, small_values_factor))
            budget_clus = int(self.max_capacity_prompt * self.kvcrush_ratio)
            # print("KVC tokens:", budget_clus)
            budget_impo = self.max_capacity_prompt - budget_clus

            # Compute page/group scores from attention scores
            intermediate_scores = attn_cache[:, :, self.start_size:]  # [bsz, num_kv_heads, intermediate_len]

            # Pad to make it divisible by page_size
            pad = intermediate_scores.shape[-1] % page_size

            if pad:
                intermediate_scores = torch.nn.functional.pad(intermediate_scores, (0, page_size - pad), mode='constant', value=0)

            # intermediate_scores = intermediate_scores[:, :, :-self.recent_size]
            # Reshape to groups and compute group scores by summing token scores within each group
            group_scores = intermediate_scores.view(bsz, num_kv_heads, -1, page_size)
            group_scores = group_scores.sum(dim=-1)  # [bsz, num_kv_heads, num_groups]

            num_recent_groups = self.recent_size // page_size
            group_scores = group_scores[:, :, :-num_recent_groups]  # Exclude recent groups

            num_intermediate_groups = group_scores.shape[-1]
            num_h2o_groups = budget_impo // page_size

            # H2O selection: topk on page scores (keep as group indices)
            h2o_group_indices = group_scores.topk(num_h2o_groups, dim=-1).indices  # [bsz, num_kv_heads, num_h2o_groups]
            # Keep h2o_group_indices in intermediate coordinates for now

            if (self.kvcrush_ratio > 0.0):
                # print("Applying KVCrush clustering")
                #kvcrush - parameters
                # print("Start size:", self.start_size)
                # print("Recent size:", self.recent_size)
                # print("Window size:", self.window_size)
                # print("Max capacity prompt:", self.max_capacity_prompt)
                # print("Sequence length:", seqlen)
                if budget_clus < page_size: #prevent edge case
                    budget_clus = page_size
                #kvcrush - rep

                # Get intermediate scores and pad them first
                normalized_scores = attn_cache[:, :, self.start_size:]
                pad = normalized_scores.shape[-1] % page_size
                if pad:
                    normalized_scores = torch.nn.functional.pad(normalized_scores, (0, page_size - pad), mode='constant', value=0)

                normalized_scores = normalized_scores[:, :, :-self.recent_size]

                # Now form binary vector from padded scores
                padded_len = normalized_scores.shape[-1]
                indices_imp = normalized_scores.topk(int(padded_len/2), dim=-1).indices
                binary_vector = torch.zeros((indices_imp.shape[0], indices_imp.shape[1], padded_len), dtype=int, device=h2o_group_indices.device)
                binary_vector[0, torch.arange(indices_imp.shape[1]).unsqueeze(1), indices_imp] = 1
                binary_vector = binary_vector.transpose(1, 2)[0]

                nrows_new = binary_vector.shape[0]
                binary_vector = binary_vector.reshape(int(nrows_new/page_size), num_kv_heads * page_size)

                #kvcrush - weak clustering
                anchor_random = torch.randint(0, 2, (1,binary_vector.shape[1]), device=h2o_group_indices.device)
                mean_point = binary_vector.float().mean(dim=0)
                anchor_invmean = (mean_point < 0.5).int()
                anchor_mean = (mean_point > 0.5).int()
                anchor_alt = torch.zeros_like(binary_vector[0])
                anchor_alt[1::2] = 1
                anchors = {
                            "inv_mean": anchor_invmean,
                            "mean": anchor_mean,
                            "alternate": anchor_alt,
                            "random": anchor_random
                            }
                dist_list = []
                distances = torch.sum(binary_vector != anchors[self.anchor_mode], dim=1)
                dist_list.append(distances)
                # Filter out groups already selected by H2O before KVCrush clustering
                # h2o_group_indices is already in intermediate coordinates [0, num_groups)
                # h2o_selected_groups = h2o_group_indices[0, :, :].flatten().unique()
                h2o_selected_groups = h2o_group_indices[0, 0, :]

                # Create group indices for intermediate region [0, num_intermediate_groups)
                all_group_idx = torch.arange(0, distances.numel(), 1, device=h2o_group_indices.device)
                mask_clus_select = ~torch.isin(all_group_idx, h2o_selected_groups)
                keep_clus_eligible = all_group_idx[mask_clus_select]

                sorted_values, indices_ = torch.sort(distances)

                sorted_dist_idx_select = torch.isin(indices_, keep_clus_eligible)
                sorted_dist_idx = indices_[sorted_dist_idx_select]

                # Only proceed with KVCrush if there are groups available (not all selected by H2O)
                if sorted_dist_idx.numel() > 0:
                    rep_indices = torch.linspace(0, sorted_dist_idx.numel() - 1, int(budget_clus / page_size), dtype=torch.long)
                    # Keep as group indices (in intermediate coordinates)
                    kvcrush_group_indices = sorted_dist_idx[rep_indices].unsqueeze(0).unsqueeze(0).expand(1, num_kv_heads, -1)
                    # Concatenate H2O and KVCrush group indices
                    keep_topk = torch.cat((h2o_group_indices, kvcrush_group_indices), dim=2)
                    # print("Picked KVCrush")
                else:
                    logger.warning(
                        "All intermediate groups already selected "
                        "by H2O, skipping KVCrush clustering")
                    keep_topk = h2o_group_indices
            else:
                # Only H2O groups
                keep_topk = h2o_group_indices

            # Sort and shift to full coordinate system (add num_past_groups offset)
            num_past_groups = self.start_size // page_size
            num_recent_groups = self.recent_size // page_size
            keep_topk = keep_topk.sort(dim=-1).values + num_past_groups

            # print(f"[KVCrush] keep_topk group indices (intermediate + KVCrush) after shifting: {keep_topk[0,0,:]}")
            # Build full group index list: [past, intermediate (H2O+KVCrush), recent]
            num_groups = num_past_groups + num_intermediate_groups + num_recent_groups
            keep_past = torch.arange(0, num_past_groups, device=keep_topk.device).unsqueeze(0).unsqueeze(0).expand(1, num_kv_heads, -1)
            keep_recent = torch.arange(num_groups - num_recent_groups, num_groups, device=keep_topk.device).unsqueeze(0).unsqueeze(0).expand(1, num_kv_heads, -1)
            keep_group_idx = torch.cat([keep_past, keep_topk, keep_recent], dim=-1)

            # Convert group indices to token indices
            indices = self._convert_group_indices(keep_group_idx, seqlen, page_size, num_kv_heads)


            #torch.set_printoptions(threshold=torch.inf)
            # print("Indices before sorting:", indices)
            # Sort indices to maintain sequential ordering of tokens
            indices, _ = torch.sort(indices, dim=-1)

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            # print("After KVCrush, indices.shape:", indices.shape)
            # Now gather from full key/value states (not just :-recent_size)
            key_states = key_states_original.gather(dim=2, index=indices)
            value_states = value_states_original.gather(dim=2, index=indices)

            return key_states, value_states
