from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

IMAGES = [
    "./examples/images/375.jpg",
]

# MODEL_NAME = "HwwwH/MiniCPM-V-2"
MODEL_NAME = "HwwwH/MiniCPM-Llama3-V-2_5_eval"

image = Image.open(IMAGES[0]).convert("RGB")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(model=MODEL_NAME,
          gpu_memory_utilization=1,
          trust_remote_code=True,
          max_model_len=4096)

messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + 'what kind of wine is this?'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]


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
