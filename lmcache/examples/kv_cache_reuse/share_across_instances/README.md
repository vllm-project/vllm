# Examples of across-instance KV cache sharing with vLLM + LMCache
LMCache should be able to reduce the generation time of the second and following calls.

We have examples for the following types of across-instance KV cache sharing:

- KV cache sharing through a centralized cache server: `centralized_sharing/`
- KV cache sharing through p2p cache transfer: `p2p_sharing/`