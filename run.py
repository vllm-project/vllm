from vllm import LLM, SamplingParams

# MODEL_NAME="neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
# MODEL_NAME="nm-testing/Phi-3-mini-128k-instruct-FP8"
# MODEL_NAME="TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
# MODEL_NAME="LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
# MODEL_NAME="facebook/opt-125m"
# MODEL_NAME="Qwen/Qwen2-57B-A14B-Instruct"
# MODEL_NAME="mistralai/Mixtral-8x7B-Instruct-v0.1"
TENSOR_PARALLEL_SIZE = 1
# MODEL_NAME="nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test"
MODEL_NAME = "nm-testing/Meta-Llama-3-8B-Instruct-W8-Channel-A8-Dynamic-Per-Token-Test"
# MODEL_NAME="neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
params = SamplingParams(temperature=0)
model = LLM(MODEL_NAME,
            enforce_eager=True,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE)
print("\n\n")
print(
    model.generate("The best thing about the internet is",
                   sampling_params=params)[0].outputs[0].text)
print("\n\n")
