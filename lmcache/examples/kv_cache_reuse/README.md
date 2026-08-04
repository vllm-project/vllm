# Examples of KV cache reusing and sharing with vLLM + LMCache
LMCache should be able to reduce the generation time of the second and following calls.

We have examples for the following types of KV cache sharing and reusing:

- KV cache reusing with local storage backends: `local_backends/`
- KV cache reusing with remote storage backends: `remote_backends/`
- KV cache sharing across different vLLM instances: `share_across_instances/`