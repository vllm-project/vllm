from vllm.prompt_adapter.request import PromptAdapterRequest

r = PromptAdapterRequest("a", 1, "a", 1)
r.__hash__()