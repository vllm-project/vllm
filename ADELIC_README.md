# Adèlic vLLM Integration (WIP)

This repository is a fork of vLLM that natively integrates the **Adèlic Dynamic Topology Router** into the PagedAttention memory engine.

By extending the `KVCacheBlock` to track $p$-adic tree topologies, and injecting a topological block-skip check into `triton_decode_attention.py`, we bypass loading physical KV blocks from SRAM entirely if their tree path does not match the current Query's routing vector. 

## Current Implementation Status
- `[x]` **Backend:** `triton_decode_attention.py` has been patched to accept `Q_Router` and `Router_Table` tensors, skipping physical blocks when a topological mismatch occurs.
- `[x]` **Cache Engine:** `KVCacheBlock` now tracks `router_indices`.
- `[ ]` **Models:** Wrap Qwen 3.6 and Gemma 4 in `vllm/model_executor/models/`.
- `[ ]` **Routing Plumbing:** Pass the `router_indices` from the prefill phase into the `single_type_kv_cache_manager.py` block allocator.

This is a work-in-progress draft integration for testing.
