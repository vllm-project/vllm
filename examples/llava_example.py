from vllm import LLM
from vllm.assets.image import ImageAsset

from PIL import Image


def rescale_image_size(image: Image.Image, size_factor: float) -> Image.Image:
    """Rescale the dimensions of an image by a constant factor."""
    new_width = int(image.width * size_factor)
    new_height = int(image.height * size_factor)
    return image.resize((new_width, new_height))


def run_llava():
    llm = LLM(
        model="llava-hf/llava-v1.6-mistral-7b-hf")  # , tensor_parallel_size=2)

    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    image = ImageAsset("stop_sign").pil_image

    # Showing image of different resolution in a batch.
    outputs = llm.generate([{
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        }
    }, {
        "prompt": prompt,
        "multi_modal_data": {
            "image": rescale_image_size(image, 0.25)
        }
    }])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_llava()
