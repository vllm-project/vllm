from io import BytesIO

import requests
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.multimodal.image import ImagePixelData

# Dynamic image input is currently not supported and therefore
# a fixed image input shape and its corresponding feature size is required.
# See https://github.com/vllm-project/vllm/pull/4199 for the complete
# configuration matrix.

llm = LLM(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    image_input_type="pixel_values",
    image_token_id=32000,
    image_input_shape="1,3,336,336",
    image_feature_size=1176,
)

prompt = "[INST] " + "<image>" * 1176 + "\nWhat is shown in this image? [/INST]"
url = "https://h2o-release.s3.amazonaws.com/h2ogpt/bigben.jpg"
image = Image.open(BytesIO(requests.get(url).content))
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": ImagePixelData(image),
    },
    sampling_params=sampling_params)

generated_text = ""
for o in outputs:
    generated_text += o.outputs[0].text

print(f"LLM output:{generated_text}")
