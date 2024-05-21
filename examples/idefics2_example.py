from vllm import LLM
from vllm.sequence import MultiModalData
import torch

if __name__ == "__main__":
    llm = LLM(
        model="HuggingFaceM4/idefics2-8b",
        image_input_type="pixel_values",
        image_token_id=32000,
        image_input_shape="1,3,980,980",
        image_feature_size=576,
        dtype='float16',
    )

    prompt = "<image>" * 64 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # This should be provided by another online or offline component.
    images = torch.load("images/flower.pt")
    print(images.shape)
    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
