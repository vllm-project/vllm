from PIL import Image

from transformers import AutoImageProcessor
from vllm.multimodal.image import ImagePixelData
from vllm import LLM, SamplingParams


IMAGES = [
    "/data1/hezhihui/vllm/examples/images/example.png",
    "/data1/hezhihui/vllm/examples/images/375.jpg"
]

MODEL_NAME = "/data1/hezhihui/projects/MiniCPM-V-2"

image = Image.open(IMAGES[1]).convert("RGB")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

llm = LLM(
    model=MODEL_NAME,
    image_input_type="pixel_values",
    image_token_id=101,
    image_input_shape="1, 3, 448, 488",
    image_feature_size=64,
    gpu_memory_utilization=0.75,
    trust_remote_code=True
)

prompt = "<用户>" + image_processor.get_slice_image_placeholder(image.size) \
        + "what kind of wine is this?" \
        + "<AI>"

sampling_params = SamplingParams(
    # temperature=0.7,
    # top_p=0.8,
    # top_k=100,
    # seed=3472,
    max_tokens=1024,
    # min_tokens=150,
    temperature=0,
    use_beam_search=True,
    # length_penalty=1.2,
    best_of=3
)


outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": ImagePixelData(image)
    },
    sampling_params=sampling_params
)
print(outputs[0].outputs[0].text)

