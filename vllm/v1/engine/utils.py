"""Engine utils"""

# Exception strings
STR_LLM_ENGINE_PROMPT_LP_APC_UNSUPPORTED = (
    "Request specifies prompt_logprobs, but prompt"
    "_logprobs are incompatible with automatic prefix caching"
    " which is currently enabled on the vLLM server. Try"
    " re-initializing LLM with enable_prefix_caching=False,"
    " or setting prompt_logprobs=None (which is the default.)")

STR_ASYNC_LLM_PROMPT_LP_APC_UNSUPPORTED = (
    "Request specifies prompt_logprobs, but prompt"
    "_logprobs are incompatible with automatic prefix caching"
    " which is currently enabled on the vLLM server. Try"
    " restarting VLLM with --no-enable-prefix-caching,"
    " or setting prompt_logprobs=None (which is the default.)")
