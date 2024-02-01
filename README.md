## Neural Magic vLLM

Fork of vLLM with sparsity.

### To Run

Clone and install magic_wand:

```bash
git clone https://github.com/neuralmagic/magic_wand.git
cd magic_wand
export TORCH_CUDA_ARCH_LIST=8.6
pip install -e .
```

Install:
```bash
cd ../
pip install -e .
```

### Run Sample

Run a 50% sparse model:

```bash
from vllm import LLM, SamplingParams

model = LLM(
    "nm-testing/Llama-2-7b-pruned50-retrained", 
    sparsity="sparse_w16a16",   # If left off, model will be loaded as dense
    enforce_eager=True,         # Does not work with cudagraphs yet
    dtype="float16",
    tensor_parallel_size=1,
    max_model_len=1024
)

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
outputs[0].outputs[0].text
```