from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 1000

# Prompts to generate completions for
prompts = [
    "Hello, my name is",
    "How are you",
    "Good morning",
    "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
]

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eos_token_id = tokenizer.eos_token_id

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    stop_token_ids=[eos_token_id],
)

# Initialize the LLM
llm = LLM(
    model=MODEL_NAME,
    enforce_eager=True,
    enable_prefix_caching=False,
)

# Generate outputs
outputs = llm.generate(prompts, sampling_params)

# Print generated results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated Text: {generated_text!r}\n")
