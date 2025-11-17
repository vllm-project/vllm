1. currently, scheduler check kv_cache, yes -> load, no -> start runnning prefill, need a way to indicate I have a cache hit but I just don't want to use it. (Adding new states)
	1. add states to indicate bidirectional loading decision in both running and waiting 
3. need to ensure the priority of requests.
	1. decode > no cache hit >  local cache hit > external cache hit 
	2. In current design, having external cache hit already makes it less favorable then the one without hitting external kv cache. But no priority given to decode.
4. need a way to load in backward, in `start_load_kv`
	1. currently assuming loading from the start with `allocated_ids` starting from the token_id where kv_cache hits. need to adopt to backward loading.
 	2. Only need to change the calling of lmcache api from vllm. The lookup in lmcache is flexible enough 
5. need to allocate blocks (speculatively) after each load complete.
	1. In current design, all blocks are allocated once for all for the loaded kv caches from lmcache right
	2. We want to allocate blocks on the go so that we can make sure we balance the one loaded and the one computed (`num_computed_blocks`) for both `waiting` and `running` requests
	3. each requests should remember its current state of the backward loading (ie. how many in the external cache hit has been loaded)
	4. also while waiting/running, `num_computed_tokens` can also increment
