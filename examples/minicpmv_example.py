from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

IMAGES = [
    "./examples/images/375.jpg",
]

MODEL_NAME = "/data1/hezhihui/openbmb/MiniCPM-V-2"
# MODEL_NAME = "/data1/hezhihui/projects/MiniCPM-Llama3-V-2_5_eval"

image = Image.open(IMAGES[0]).convert("RGB")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(model=MODEL_NAME,
          gpu_memory_utilization=1,
          trust_remote_code=True,
          max_model_len=4096)
# 2.0
prompt = "<用户>(<image>./</image>)what kind of wine is this?<AI>"
stop_token_ids = [tokenizer.eos_id]
# 2.5
# stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
# prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + \
#         "(<image>./</image>)\nwhat kind of wine is this?" + \
#         "<|eot_id|>" + \
#         "<|start_header_id|>assistant<|end_header_id|>\n\n"

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    # temperature=0.7,
    # top_p=0.8,
    # top_k=100,
    # seed=3472,
    max_tokens=1024,
    # min_tokens=150,
    temperature=0,
    use_beam_search=True,
    # length_penalty=1.2,
    best_of=3)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    }
}, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
