from transformers import LlavaNextImageProcessor
from PIL import Image

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import ImagePixelData

IMAGES = [
    "/data1/hezhihui/vllm/examples/images/example.png",
    "/data1/hezhihui/vllm/examples/images/stop_sign.jpg"
]

MODEL_NAME = "llava-hf/llava-v1.6-34b-hf"
IMAGE_HEIGHT = IMAGE_WIDTH = 560

model_config = ModelConfig(
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    tokenizer_mode="auto",
    trust_remote_code=False,
    seed=0,
    dtype="auto",
    revision=None,
)
vlm_config = VisionLanguageConfig(
    image_input_type=VisionLanguageConfig.ImageInputType.PIXEL_VALUES,
    image_token_id=64000,
    image_input_shape=(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH),
    image_feature_size=2928,
    image_processor=MODEL_NAME,
    image_processor_revision=None,
)

image_precessor = LlavaNextImageProcessor.from_pretrained(MODEL_NAME)

image = Image.open(IMAGES[0]).convert("RGB")
image2 = Image.open(IMAGES[1]).convert("RGB")

print(image2.size)
hf_result = image_precessor.preprocess(image2, return_tensors="pt")
print(hf_result["pixel_values"].dtype)
print(hf_result["image_sizes"].shape)

vllm_result = MULTIMODAL_REGISTRY.process_input(
    ImagePixelData(image),
    model_config=model_config,
    vlm_config=vlm_config
)

print(hf_result.keys(), vllm_result.keys())
print(vllm_result["pixel_values"].shape)
