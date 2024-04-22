
from vllm import LLM
from vllm.sequence import MultiModalData

llm = LLM(
    model="HuggingFaceM4/idefics2-8b",
    image_input_type="pixel_values",
    image_token_id=32000,
    image_input_shape="1,3,336,336",
    image_feature_size=576,
)

print("passed")