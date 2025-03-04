(auto-awq)=

# AutoAWQ

To create a new 4-bit quantized model, you can leverage [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).
Quantizing reduces the model's precision from FP16 to INT4 which effectively reduces the file size by ~70%.
The main benefits are lower latency and memory usage.

You can quantize your own models by installing AutoAWQ or picking one of the [6500+ models on Huggingface](https://huggingface.co/models?sort=trending&search=awq).

```console
pip install autoawq
```

After installing AutoAWQ, you are ready to quantize a model. Please refer to the `AutoAWQ documentation <https://casper-hansen.github.io/AutoAWQ/examples/#basic-quantization>`_ for further details. Here is an example of how to quantize `mistralai/Mistral-7B-Instruct-v0.2`:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
quant_path = 'mistral-instruct-v0.2-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

To run an AWQ model with vLLM, you can use [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ) with the following command:

```console
python examples/offline_inference/llm_engine_example.py --model TheBloke/Llama-2-7b-Chat-AWQ --quantization awq
```

AWQ models are also supported directly through the LLM entrypoint:

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
