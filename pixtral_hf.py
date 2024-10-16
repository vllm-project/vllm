from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_base64
import torch
torch.set_printoptions(sci_mode=False)

sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

def image_url(asset: str):
    image = ImageAsset(asset)
    base64 = encode_image_base64(image.pil_image)
    return f"data:image/jpeg;base64,{base64}"


def reference_one_image():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    model_id = "mistral-community/pixtral-12b"
    model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto").to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)

    IMGS = [
        ImageAsset("stop_sign").pil_image.convert("RGB"),
    ]
    PROMPT = f"<s>[INST][IMG]Describe the image.[/INST]"

    inputs = processor(text=PROMPT, images=IMGS, return_tensors="pt").to(model.device)
    print("inputs['input_ids']", inputs["input_ids"].shape)
    print("detok(inputs['input_ids'])", processor.batch_decode(inputs["input_ids"])[0])
    print("inputs['pixel_values']", [i.shape for i in inputs["pixel_values"]])
    generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)
    """
    The image features a beautiful cherry blossoms in foreground, creating a picturesque frame around the Washington Monument. The monument is illuminated with a warm, golden light, standing tall against a clear, blue sky. The blossoms are in full bloom, with delicate, pink petals, adding a soft, romantic touch to the scene. The perspective is from a low angle, looking up at the monument, emphasizing its height and grandeur. The overall mood of the image is serene and peaceful, celebrating the
    """

def pixtralhf_one_image():
    model_name = "mistral-community/pixtral-12b"
    llm = LLM(
        model=model_name, 
        max_num_seqs=1, 
        enforce_eager=True, 
        max_model_len=10000, 
    )

    chat_template = "{%- if messages[0][\"role\"] == \"system\" %}\n    {%- set system_message = messages[0][\"content\"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message[\"role\"] == \"user\" %}\n        {%- if loop.last and system_message is defined %}\n            {{- \"[INST]\" + system_message + \"\n\n\" }}\n        {%- else %}\n            {{- \"[INST]\" }}\n        {%- endif %}\n        {%- if message[\"content\"] is not string %}\n            {%- for chunk in message[\"content\"] %}\n                {%- if chunk[\"type\"] == \"text\" %}\n                    {{- chunk[\"content\"] }}\n                {%- elif chunk[\"type\"] == \"image\" %}\n                    {{- \"[IMG]\" }}\n                {%- else %}\n                    {{- raise_exception(\"Unrecognized content type!\") }}\n                {%- endif %}\n            {%- endfor %}\n        {%- else %}\n            {{- message[\"content\"] }}\n        {%- endif %}\n        {{- \"[/INST]\" }}\n    {%- elif message[\"role\"] == \"assistant\" %}\n        {{- message[\"content\"] + eos_token}}\n    {%- else %}\n        {{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}\n    {%- endif %}\n{%- endfor %}"

    # image1 = ImageAsset("stop_sign").pil_image.convert("RGB")
    # inputs = {
    #     "prompt": "<s>[INST][IMG]Describe the image.[/INST]",
    #     "multi_modal_data": {"image": image1},
    # }
    # outputs = llm.generate(inputs, sampling_params=sampling_params)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url("stop_sign")
                    }
                },
            ],
        },
    ]
    outputs = llm.chat(messages, sampling_params=sampling_params, chat_template=chat_template)

    print(outputs[0].outputs[0].text)
    """
    This image appears to be a close-up view of a large number of pink flowers, possibly cherry blossoms, against a blue sky background. The flowers are densely packed and fill the entire frame of the image, creating a vibrant and colorful display. The blue sky provides a striking contrast to the pink flowers, enhancing their visual appeal. The image does not contain any discernible text or other objects. It focuses solely on the flowers and the sky, capturing a moment of natural beauty.
    """

def pixtral_one_image():
    model_name = "mistralai/Pixtral-12B-2409"
    llm = LLM(
        model=model_name, 
        max_num_seqs=1, 
        enforce_eager=True, 
        max_model_len=10000,
        tokenizer_mode="mistral",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url("stop_sign")
                    }
                },
            ],
        },
    ]
    outputs = llm.chat(messages, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)
    """
    This image appears to be a close-up view of a large number of pink flowers, possibly cherry blossoms, against a blue sky background. The flowers are densely packed and fill the entire frame of the image, creating a vibrant and colorful display. The blue sky provides a striking contrast to the pink flowers, enhancing their visual appeal. The image does not contain any discernible text or other objects. It focuses solely on the flowers and the sky, capturing a moment of natural beauty.
    """

def pixtralhf_two_image():
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
        "prompt": "<s>[INST][IMG][IMG]Describe the images.[/INST]",
        "multi_modal_data": {
            "image": [image1, image2]
        },
    }
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)

def pixtralhf_fp8_one_image():
    model_name = "nm-testing/pixtral-12b-FP8-dynamic"
    llm = LLM(
        model=model_name, 
        max_num_seqs=1, 
        enforce_eager=True, 
        max_model_len=10000, 
    )

    image1 = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    inputs = {
        "prompt": "<s>[INST][IMG]Describe the image.[/INST]",
        "multi_modal_data": {"image": image1},
    }
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)
    """
    This image appears to be a close-up view of a large number of pink flowers, possibly cherry blossoms, against a blue sky background. The flowers are densely packed and fill the entire frame of the image. The vibrant pink color of the flowers contrasts beautifully with the clear blue sky, creating a visually striking scene. The image likely captures a moment of natural beauty, possibly during the spring season when cherry blossoms are in full bloom.
    """

# reference_one_image()
pixtralhf_one_image()
# pixtral_one_image()

# pixtralhf_two_image()
