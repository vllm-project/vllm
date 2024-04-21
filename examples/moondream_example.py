import torch
from PIL import Image
from torchvision.transforms.v2 import (Compose, InterpolationMode, Normalize,
                                       Resize, ToDtype, ToImage)

from vllm import LLM, SamplingParams
from vllm.sequence import MultiModalData

if __name__ == "__main__":

    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    llm = LLM(
        model="vikhyatk/moondream2",
        trust_remote_code=True,
        image_input_type="pixel_values",
        image_token_id=50256,
        image_input_shape="1,3,378,378",
        image_feature_size=729,
    )

    preprocess = Compose([
        Resize(size=(378, 378), interpolation=InterpolationMode.BICUBIC),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image = Image.open("docs/source/assets/kernel/value.png").convert("RGB")
    image_pixels = preprocess(image).unsqueeze(0)

    outputs = llm.generate(
        [("<|endoftext|>" * 729) +
         "\n\nQuestion: Describe this image.\n\nAnswer:"],
        multi_modal_data=MultiModalData(type=MultiModalData.Type.IMAGE,
                                        data=image_pixels),
        sampling_params=sampling_params,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
