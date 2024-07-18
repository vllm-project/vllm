from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

sample_params = SamplingParams(temperature=0, max_tokens=1024)
model = "deepseek-ai/deepseek-vl-7b-chat"
model = "deepseek-ai/deepseek-vl-1.3b-chat"
prompt_one = (
    "You are a helpful language and vision assistant."
    "You are able to understand the visual content that the user provides,"
    "and assist the user with a variety of tasks using natural language.\n"
    "User: <image_placeholder> Describe the content of this image.\nAssistant:"
)

prompt_two = (
    "You are a helpful language and vision assistant. You are able to "
    "understand the visual content that the user provides, and assist the "
    "user with a variety of tasks using natural language.\n User: "
    "<image_placeholder>What is the season?\nAssistant:")


def run_deepseek_vl():
    llm = LLM(model=model)
    stop_sign_image = ImageAsset("stop_sign").pil_image
    cherry_blossom_image = ImageAsset("cherry_blossom").pil_image
    outputs = llm.generate(
        [
            {
                "prompt": prompt_one,
                "multi_modal_data": {
                    "image": stop_sign_image
                },
            },
            {
                "prompt": prompt_two,
                "multi_modal_data": {
                    "image": cherry_blossom_image
                },
            },
        ],
        sampling_params=sample_params,
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print("------------------")
        print(generated_text)


def main():
    run_deepseek_vl()


if __name__ == "__main__":
    main()
