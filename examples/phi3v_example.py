from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def run_phi3v():
    model_path = "microsoft/Phi-3-vision-128k-instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=5,
    )

    image = ImageAsset("cherry_blossom").pil_image

    # single-image prompt
    prompt = "<|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n"  # noqa: E501
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        },
        sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_phi3v()
