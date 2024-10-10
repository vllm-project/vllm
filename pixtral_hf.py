from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

def reference_one_image():
    model_name = "mistral-community/pixtral-12b"
    llm = LLM(
        model=model_name, 
        max_num_seqs=1, 
        enforce_eager=True, 
        max_model_len=10000, 
    )

    image1 = ImageAsset("stop_sign").pil_image.convert("RGB")
    inputs = {
        "prompt": f"<s>[INST]Describe the image.\n[IMG][/INST]",
        "multi_modal_data": {"image": image1},
    }
    outputs = llm.generate(inputs, sampling_params=SamplingParams(temperature=0.0, max_tokens=100))

    print(outputs[0].outputs[0].text)
    """
    This image appears to be a close-up view of a large number of pink flowers, possibly cherry blossoms, against a blue sky background. The flowers are densely packed and fill the entire frame of the image, creating a vibrant and colorful display. The blue sky provides a striking contrast to the pink flowers, enhancing their visual appeal. The image does not contain any discernible text or other objects. It focuses solely on the flowers and the sky, capturing a moment of natural beauty.
    """

def reference_two_image():
    model_name = "mistral-community/pixtral-12b"
    llm = LLM(
        model=model_name, 
        max_num_seqs=1, 
        enforce_eager=True, 
        max_model_len=10000, 
        limit_mm_per_prompt={"image": 2}
    )

    image1 = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    image2 = ImageAsset("stop_sign").pil_image.convert("RGB")
    inputs = {
        "prompt": f"<s>[INST]Describe the images.\n[IMG][IMG][/INST]",
        "multi_modal_data": {
            "image": [image1, image2]
        },
    }
    outputs = llm.generate(inputs, sampling_params=SamplingParams(temperature=0.0, max_tokens=100))

    print(outputs[0].outputs[0].text)

def fp8_one_image():
    model_name = "nm-testing/pixtral-12b-FP8-dynamic"
    llm = LLM(
        model=model_name, 
        max_num_seqs=1, 
        enforce_eager=True, 
        max_model_len=10000, 
    )

    image1 = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    inputs = {
        "prompt": f"<s>[INST]Describe the image.\n[IMG][/INST]",
        "multi_modal_data": {"image": image1},
    }
    outputs = llm.generate(inputs, sampling_params=SamplingParams(temperature=0.0, max_tokens=100))

    print(outputs[0].outputs[0].text)
    """
    This image appears to be a close-up view of a large number of pink flowers, possibly cherry blossoms, against a blue sky background. The flowers are densely packed and fill the entire frame of the image. The vibrant pink color of the flowers contrasts beautifully with the clear blue sky, creating a visually striking scene. The image likely captures a moment of natural beauty, possibly during the spring season when cherry blossoms are in full bloom.
    """

reference_one_image()
# reference_two_image()
