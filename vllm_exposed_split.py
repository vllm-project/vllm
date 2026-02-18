import os
import shutil
from pathlib import Path

# Set environment variables
os.environ["TORCH_TRACE"] = "/workspace/torch_trace"
os.environ["TORCH_LOGS"] = "output_code"
os.environ["VLLM_MLA_EXPOSED_SPLIT"] = "0"

# Clear compilation cache to force recompilation
vllm_cache = Path.home() / ".cache" / "vllm"
torchinductor_cache = Path(f"/tmp/torchinductor_{os.getenv('USER', 'unknown')}")
ddpath = Path("/workspace/ddpath")
if vllm_cache.exists():
    shutil.rmtree(vllm_cache)
    print(f"Cleared vLLM cache: {vllm_cache}")
if torchinductor_cache.exists():
    shutil.rmtree(torchinductor_cache)
    print(f"Cleared torchinductor cache: {torchinductor_cache}")
if ddpath.exists():
    shutil.rmtree(ddpath)
    print(f"Cleared ddpath: {ddpath}")

# Import vllm first to avoid import issues
from vllm import LLM, SamplingParams
from vllm.config.compilation import (
    CompilationConfig, CompilationMode, CUDAGraphMode,
    DynamicShapesConfig, DynamicShapesType,
)

if __name__ == '__main__':
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        max_model_len=256,
        trust_remote_code=True,
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            use_inductor_graph_partition=True,
            debug_dump_path="/workspace/ddpath"
        ),
    )

    prompts = ["The capital of France is"]
    out = llm.generate(prompts, SamplingParams(max_tokens=20, temperature=0))
    for o in out:
        print(f"Prompt: {o.prompt!r}")
        print(f"Output: {o.outputs[0].text!r}")
        print()
