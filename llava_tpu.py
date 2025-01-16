from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# Initialize LLaVA model
llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    max_model_len=2048,
    max_num_seqs=2,
    dtype="bfloat16",
)

# Load sample image
image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

# Create two different prompts
prompts = [
    "What do you see in this image?",
    "What colors are most prominent in this image?",
]

# Format prompts according to LLaVA's requirements
formatted_inputs = [
    {
        "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
        "multi_modal_data": {"image": image}
    }
    for prompt in prompts
]

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)

# Generate responses
outputs = llm.generate(formatted_inputs, sampling_params=sampling_params)

# Print results
for i, output in enumerate(outputs):
    print(f"\nPrompt {i + 1}: {prompts[i]}")
    print(f"Response: {output.outputs[0].text}")
