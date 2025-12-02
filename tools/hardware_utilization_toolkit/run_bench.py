import argparse
import asyncio
import os
from transformers import AutoTokenizer
  
from vllm import AsyncLLMEngine, SamplingParams
from vllm.entrypoints.api_server import AsyncEngineArgs
from typing import Optional 
from vllm.config import CompilationConfig
  
################################ Helper Function ################################
def make_compilation_config(
        cuda_graph_mode: str = "PIECEWISE",
        compile_sizes_list: Optional[str] = None
) -> CompilationConfig:
    """
    Build a CompilationConfig from provided values.
    """
    kwargs = {"cudagraph_mode": cuda_graph_mode}
 
    if compile_sizes_list:
        kwargs["compile_sizes"] = [int(size) for size in compile_sizes_list.split(",")]
 
    return CompilationConfig(**kwargs)
 
################################ Build Customized N Length Prompt################################
 
# Parse CLI args
parser = argparse.ArgumentParser(description="Run vLLM with customized length prompts.")
parser.add_argument('--target_length', type=int, default=31744, help='Target prompt token length')
parser.add_argument('--output_length', type=int, default=700, help='Output token length')
args = parser.parse_args()
 
# Set desired prompt token length here (e.g., 6k, 8k)
TARGET_PROMPT_TOKEN_LENGTH = args.target_length
OUTPUT_TOKEN_LENGTH = args.output_length
 
#Get rid of nile_triton
 
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "/data/neolite_bf16_ckpt/")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/neolite_bf16_ckpt/")
GPU_VLLM_BLOCK_SIZE = int(os.getenv("GPU_VLLM_BLOCK_SIZE", 16))
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", "False").lower() in ["true", "t", "1"]
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", 8))
VLLM_SWAP_SPACE = 4
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("MAX_NUM_BATCHED_TOKENS", 131072))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "16"))
 
 
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
 
# Base: informative technical content
intro = (
    "Title: The State of Artificial Intelligence in 2025\n\n"
    "Abstract: This report provides an overview of major advances in artificial intelligence as of the year 2025. "
    "It covers developments in natural language models, reinforcement learning, multimodal systems, and ethical challenges. "
    "We also provide a critical look at the deployment of large language models in commercial applications and their implications on society.\n\n"
)
 
# Simulated long-form content (use coherent technical paragraphs)
base_paragraph = (
    "Large language models (LLMs) have demonstrated remarkable performance in a variety of NLP tasks including summarization, translation, and question answering. "
    "Techniques like retrieval-augmented generation (RAG), parameter-efficient fine-tuning, and speculative decoding are widely adopted. "
    "At the same time, challenges remain in model alignment, hallucination control, and latency under constrained inference budgets. "
    "In production environments, inference optimizations such as KV cache reuse, continuous batching, and CUDA Graph integration are critical for cost-effective deployment. "
    "Moreover, foundation models are increasingly evaluated not just on accuracy but also on robustness, fairness, and interpretability. "
)
 
# Compose long prompt
prompt_text = intro
while len(tokenizer(prompt_text).input_ids) < TARGET_PROMPT_TOKEN_LENGTH - 100:
    prompt_text += base_paragraph
 
# Add a final instruction for generation
prompt_text += (
    "\n\n---\n\n"
    "Based on the content above, please summarize the key challenges facing large-scale model inference in production, "
    "and propose potential solutions to reduce latency and cost while preserving model quality."
)
 
# Trim to exactly TARGET_PROMPT_TOKEN_LENGTH tokens
input_ids = tokenizer(prompt_text).input_ids[:TARGET_PROMPT_TOKEN_LENGTH]
final_prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
 
# Inject into vLLM script
prompts = [final_prompt, final_prompt]
request_ids = ["0", "1"]
 
# Optional preview
print(f"✅ Prompt token count: {len(tokenizer(final_prompt).input_ids)}")
print(f"📝 Prompt preview:\n{final_prompt[:1000]}")
 
CUDA_GRAPH_MODE = str(os.environ.get("CUDA_GRAPH_MODE", "PIECEWISE"))
# https://code.amazon.com/packages/SFAI-VLLM/blobs/44ddda9bc54b97537d3f5b279577a8184372c7e7/--/vllm/config/compilation.py#L41
COMPILE_SIZES_LIST = os.environ.get("COMPILE_SIZES_LIST", None)
USE_DUMMY_WEIGHT = str(os.environ.get("USE_DUMMY_WEIGHT", "False")).lower() in ["true", "t", "1"]
USE_NEOLITE_MODEL = str(os.environ.get("USE_NEOLITE_MODEL", "False")).lower() in ["true", "t", "1"]
 
kwargs = dict(
    model=MODEL_PATH,
    tokenizer=TOKENIZER_PATH,
    tokenizer_mode="auto",
    seed=0,
    max_num_seqs=MAX_BATCH_SIZE,
    max_model_len=131072,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    swap_space=VLLM_SWAP_SPACE,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    trust_remote_code=False,
    disable_log_stats=False,
    dtype="auto",
    enable_prefix_caching=ENABLE_PREFIX_CACHING,
    kv_cache_dtype=KV_CACHE_DTYPE,
    block_size=GPU_VLLM_BLOCK_SIZE,
    enable_chunked_prefill=True,
    collect_time_per_step=False,
    compilation_config=make_compilation_config(cuda_graph_mode=CUDA_GRAPH_MODE, compile_sizes_list=COMPILE_SIZES_LIST),
)
if USE_DUMMY_WEIGHT:
    kwargs["load_format"] = "dummy"
 
if USE_NEOLITE_MODEL:
    kwargs["hf_overrides"] = {"architectures": ["NeoLiteMixtralForCausalLM"], "kv_lora_rank": None}
 
engine_args = AsyncEngineArgs(
    **kwargs,
)
print(f"engine_args is {engine_args}")
 
 
async def generate_all_requests(prompts, request_ids):
    print("🚀 Sending prompts to vLLM engine...")
    print("Request_Id: ", request_ids)
    async_llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        top_k=1,
        max_tokens=OUTPUT_TOKEN_LENGTH,
        stop_token_ids=[int(token) for token in os.getenv("STOP_TOKEN_IDS", "199999,200002").split(",")]
    )
 
    if USE_NEOLITE_MODEL:
        print("Updating sampling params for NeoLite...")
        sampling_params = SamplingParams(temperature=0.6,
                                        top_p=1.0, # same as default value in SamplingParams
                                        top_k=-1, 
                                        min_p=0.0,  # same as default value in SamplingParams
                                        stop="<|im_end|>",
                                        stop_token_ids=[200002],
                                        n=1, # same as default value in SamplingParams
                                        ignore_eos=False,  # same as default value in SamplingParams
                                        guided_decoding=None,  # same as default value in SamplingParams
                                        max_tokens=OUTPUT_TOKEN_LENGTH, # temp value, update after final optimization
                                        )
 
    print(f"sampling params is {sampling_params}")
 
    async def process_single_prompt(prompt, request_id):
        try:
            generated_text = ""
            async for output in async_llm_engine.generate(prompt, sampling_params, request_id):
                generated_text = output.outputs[0].text
            print(f"\n{'='*80}")
            print(f"🆔 Request ID: {request_id}")
            print(f"\n📥 Prompt preview (first 100 chars):\n{prompt[:100]}...")
            print(f"\n📤 Generated Text:\n{generated_text}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"❌ Error for prompt {request_id}: {str(e)}")
 
    # Create tasks for all prompts
    tasks = [process_single_prompt(prompt, req_id) for prompt, req_id in zip(prompts, request_ids)]
 
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
 
 
# Run the async function
asyncio.run(generate_all_requests(prompts, request_ids))
 