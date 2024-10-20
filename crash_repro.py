import contextlib
import weakref

import pytest

import torch

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment


# anonomyzed sequence from the user requests that caused crash in production
# causes illegal memory access with block manager v2, chunked prefill and
# prefix caching (and maybe sliding window?)
# consistenly reproduces on 1xP40
PROMPTS = [
  TokensPrompt(prompt_token_ids=([0] * 588 ) + ([1] * 1332) + ([2] * 30  ) + ([3] * 1   )),
  TokensPrompt(prompt_token_ids=([0] * 588 ) + ([1] * 1332) + ([4] * 3   ) + ([5] * 50  )),
  TokensPrompt(prompt_token_ids=([0] * 588 ) + ([1] * 1332) + ([2] * 30  ) + ([6] * 95  )),
  TokensPrompt(prompt_token_ids=([0] * 588 ) + ([1] * 1332) + ([4] * 3   ) + ([7] * 174 )),
  TokensPrompt(prompt_token_ids=([0] * 588 ) + ([8] * 1539)                              ),
]


@pytest.fixture
def llm(enable_chunked_prefill, enable_prefix_caching, use_v2_block_manager, monkeypatch):
  if not use_v2_block_manager:
    monkeypatch.setenv("VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1", "1")

  llm = LLM(
    config_format="mistral",
    dtype="float16",
    enable_chunked_prefill=enable_chunked_prefill,
    enable_prefix_caching=enable_prefix_caching,
    enforce_eager=True,
    load_format="mistral",
    max_model_len=4096,
    model="sasha0552/Ministral-8B-Instruct-2410",
    swap_space=0,
    tensor_parallel_size=1,
    tokenizer_mode="mistral",
    use_v2_block_manager=use_v2_block_manager,
  )

  with llm.deprecate_legacy_api():
    yield weakref.proxy(llm)

    del llm

  destroy_model_parallel()
  destroy_distributed_environment()
  with contextlib.suppress(AssertionError):
    torch.distributed.destroy_process_group()
  try:
    torch.cuda.empty_cache()
  except:
    # idk what to do to reset pytorch after illegal memory access
    pass

  if not use_v2_block_manager:
    monkeypatch.delenv("VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1")

# v1 block manager is not available on main branch
@pytest.mark.parametrize("use_v2_block_manager,enable_chunked_prefill,enable_prefix_caching,crash_expected", [
  #(False, False, False, False), # v1
  ( True, False, False, False), # v2
  #(False,  True, False, False), # v1 + chunked prefill
  ( True,  True, False, False), # v2 + chunked prefill
  #(False, False,  True, False), # v1 + prefix caching
  ( True, False,  True, False), # v2 + prefix caching
  #(False,  True,  True, False), # v1 + chunked prefill + prefix caching
  ( True,  True,  True,  True), # v2 + chunked prefill + prefix caching
])
def test_vllm(crash_expected, llm):
  # process prompts one by one to avoid batching
  for i, prompt in enumerate(PROMPTS):
    if crash_expected and i == 4:
      with pytest.raises(RuntimeError, match=r".*CUDA error: an illegal memory access was encountered.*"):
        llm.generate(prompt, SamplingParams(max_tokens=1))
    else:
      llm.generate(prompt, SamplingParams(max_tokens=1))

  del llm
