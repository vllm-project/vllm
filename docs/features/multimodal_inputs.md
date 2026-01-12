# Multimodal Inputs

This page teaches you how to pass multi-modal inputs to [multi-modal models](../models/supported_models.md#list-of-multimodal-language-models) in vLLM.

!!! note
    We are actively iterating on multi-modal support. See [this RFC](https://github.com/vllm-project/vllm/issues/4194) for upcoming changes,
    and [open an issue on GitHub](https://github.com/vllm-project/vllm/issues/new/choose) if you have any feedback or feature requests.

!!! tip
    When serving multi-modal models, consider setting `--allowed-media-domains` to restrict domain that vLLM can access to prevent it from accessing arbitrary endpoints that can potentially be vulnerable to Server-Side Request Forgery (SSRF) attacks. You can provide a list of domains for this arg. For example: `--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`

    Also, consider setting `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` to prevent HTTP redirects from being followed to bypass domain restrictions.

    This restriction is especially important if you run vLLM in a containerized environment where the vLLM pods may have unrestricted access to internal networks.

## Offline Inference

To input multi-modal data, follow this schema in [vllm.inputs.PromptType][]:

- `prompt`: The prompt should follow the format that is documented on HuggingFace.
- `multi_modal_data`: This is a dictionary that follows the schema defined in [vllm.multimodal.inputs.MultiModalDataDict][].

### Stable UUIDs for Caching (multi_modal_uuids)

When using multi-modal inputs, vLLM normally hashes each media item by content to enable caching across requests. You can optionally pass `multi_modal_uuids` to provide your own stable IDs for each item so caching can reuse work across requests without rehashing the raw content.

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL example with two images
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_a = Image.open("/path/to/a.jpg")
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [img_a, img_b]},
        # Provide stable IDs for caching.
        # Requirements (matched by this example):
        #  - Include every modality present in multi_modal_data.
        #  - For lists, provide the same number of entries.
        #  - Use None to fall back to content hashing for that item.
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

Using UUIDs, you can also skip sending media data entirely if you expect cache hits for respective items. Note that the request will fail if the skipped media doesn't have a corresponding UUID, or if the UUID fails to hit the cache.

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL example with two images
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [None, img_b]},
        # Since img_a is expected to be cached, we can skip sending the actual
        # image entirely.
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

!!! warning
    If both multimodal processor caching and prefix caching are disabled, user-provided `multi_modal_uuids` are ignored.

### Image Inputs

You can pass a single image to the `'image'` field of the multi-modal dictionary, as shown in the following examples:

??? code

    ```python
    from vllm import LLM

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # Load the image using PIL.Image
    image = PIL.Image.open(...)

    # Single prompt inference
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # Batch inference
    image_1 = PIL.Image.open(...)
    image_2 = PIL.Image.open(...)
    outputs = llm.generate(
        [
            {
                "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_1},
            },
            {
                "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_2},
            }
        ]
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

Full example: [examples/offline_inference/vision_language.py](../../examples/offline_inference/vision_language.py)

To substitute multiple images inside the same text prompt, you can pass in a list of images instead:

??? code

    ```python
    from vllm import LLM

    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
        limit_mm_per_prompt={"image": 2},  # The maximum number to accept
    )

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"

    # Load the images using PIL.Image
    image1 = PIL.Image.open(...)
    image2 = PIL.Image.open(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [image1, image2]},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

Full example: [examples/offline_inference/vision_language_multi_image.py](../../examples/offline_inference/vision_language_multi_image.py)

If using the [LLM.chat](../models/generative_models.md#llmchat) method, you can pass images directly in the message content using various formats: image URLs, PIL Image objects, or pre-computed embeddings:

??? code

    ```python
    from vllm import LLM
    from vllm.assets.image import ImageAsset

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    image_url = "https://picsum.photos/id/32/512/512"
    image_pil = ImageAsset('cherry_blossom').pil_image
    image_embeds = torch.load(...)

    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "image_pil",
                    "image_pil": image_pil,
                },
                {
                    "type": "image_embeds",
                    "image_embeds": image_embeds,
                },
                {
                    "type": "text",
                    "text": "What's in these images?",
                },
            ],
        },
    ]

    # Perform inference and log output.
    outputs = llm.chat(conversation)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

Multi-image input can be extended to perform video captioning. We show this with [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) as it supports videos:

??? code

    ```python
    from vllm import LLM

    # Specify the maximum number of frames per video to be 4. This can be changed.
    llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})

    # Create the request payload.
    video_frames = ... # load your video making sure it only has the number of frames specified earlier.
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this set of frames. Consider the frames to be a part of the same video.",
            },
        ],
    }
    for i in range(len(video_frames)):
        base64_image = encode_image(video_frames[i]) # base64 encoding.
        new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        message["content"].append(new_image)

    # Perform inference and log output.
    outputs = llm.chat([message])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

#### Custom RGBA Background Color

When loading RGBA images (images with transparency), vLLM converts them to RGB format. By default, transparent pixels are replaced with white background. You can customize this background color using the `rgba_background_color` parameter in `media_io_kwargs`.

??? code

    ```python
    from vllm import LLM

    # Default white background (no configuration needed)
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # Custom black background for dark theme
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 0]}},
    )

    # Custom brand color background (e.g., blue)
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 255]}},
    )
    ```

!!! note
    - The `rgba_background_color` accepts RGB values as a list `[R, G, B]` or tuple `(R, G, B)` where each value is 0-255
    - This setting only affects RGBA images with transparency; RGB images are unchanged
    - If not specified, the default white background `(255, 255, 255)` is used for backward compatibility

### Video Inputs

You can pass a list of NumPy arrays directly to the `'video'` field of the multi-modal dictionary
instead of using multi-image input.

Instead of NumPy arrays, you can also pass `'torch.Tensor'` instances, as shown in this example using Qwen2.5-VL:

??? code

    ```python
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    video_path = "https://content.pexels.com/videos/free-videos.mp4"

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(max_tokens=1024)

    video_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this video."},
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]

    messages = video_messages
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

    !!! note
        'process_vision_info' is only applicable to Qwen2.5-VL and similar models.

Full example: [examples/offline_inference/vision_language.py](../../examples/offline_inference/vision_language.py)

### Audio Inputs

You can pass a tuple `(array, sampling_rate)` to the `'audio'` field of the multi-modal dictionary.

Full example: [examples/offline_inference/audio_language.py](../../examples/offline_inference/audio_language.py)

#### Automatic Audio Channel Normalization

vLLM automatically normalizes audio channels for models that require specific audio formats. When loading audio with libraries like `torchaudio`, stereo files return shape `[channels, time]`, but many audio models (particularly Whisper-based models) expect mono audio with shape `[time]`.

**Supported models with automatic mono conversion:**

- **Whisper** and all Whisper-based models
- **Qwen2-Audio**
- **Qwen2.5-Omni** / **Qwen3-Omni** (inherits from Qwen2.5-Omni)
- **Ultravox**

For these models, vLLM automatically:

1. Detects if the model requires mono audio via the feature extractor
2. Converts multi-channel audio to mono using channel averaging
3. Handles both `(channels, time)` format (torchaudio) and `(time, channels)` format (soundfile)

**Example with stereo audio:**

```python
import torchaudio
from vllm import LLM

# Load stereo audio file - returns (channels, time) shape
audio, sr = torchaudio.load("stereo_audio.wav")
print(f"Original shape: {audio.shape}")  # e.g., torch.Size([2, 16000])

# vLLM automatically converts to mono for Whisper-based models
llm = LLM(model="openai/whisper-large-v3")

outputs = llm.generate({
    "prompt": "",
    "multi_modal_data": {"audio": (audio.numpy(), sr)},
})
```

No manual conversion is needed - vLLM handles the channel normalization automatically based on the model's requirements.

### Embedding Inputs

To input pre-computed embeddings belonging to a data type (i.e. image, video, or audio) directly to the language model,
pass a tensor of shape `(num_items, feature_size, hidden_size of LM)` to the corresponding field of the multi-modal dictionary.

You must enable this feature via `enable_mm_embeds=True`.

!!! warning
    The vLLM engine may crash if incorrect shape of embeddings is passed.
    Only enable this flag for trusted users!

#### Image Embeddings

??? code

    ```python
    from vllm import LLM

    # Inference with image embeddings as input
    llm = LLM(model="llava-hf/llava-1.5-7b-hf", enable_mm_embeds=True)

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # Embeddings for single image
    # torch.Tensor of shape (1, image_feature_size, hidden_size of LM)
    image_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

For Qwen2-VL and MiniCPM-V, we accept additional parameters alongside the embeddings:

??? code

    ```python
    # Construct the prompt based on your model
    prompt = ...

    # Embeddings for multiple images
    # torch.Tensor of shape (num_images, image_feature_size, hidden_size of LM)
    image_embeds = torch.load(...)

    # Qwen2-VL
    llm = LLM(
        "Qwen/Qwen2-VL-2B-Instruct",
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            "image_embeds": image_embeds,
            # image_grid_thw is needed to calculate positional encoding.
            "image_grid_thw": torch.load(...),  # torch.Tensor of shape (1, 3),
        }
    }

    # MiniCPM-V
    llm = LLM(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            "image_embeds": image_embeds,
            # image_sizes is needed to calculate details of the sliced image.
            "image_sizes": [image.size for image in images],  # list of image sizes
        }
    }

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": mm_data,
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

For Qwen3-VL, the `image_embeds` should contain both the base image embedding and deepstack features.

#### Audio Embedding Inputs

You can pass pre-computed audio embeddings similar to image embeddings:

??? code

    ```python
    from vllm import LLM
    import torch

    # Enable audio embeddings support
    llm = LLM(model="fixie-ai/ultravox-v0_5-llama-3_2-1b", enable_mm_embeds=True)

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <audio>\nWhat is in this audio?\nASSISTANT:"

    # Load pre-computed audio embeddings
    # torch.Tensor of shape (1, audio_feature_size, hidden_size of LM)
    audio_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"audio": audio_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

## Online Serving

Our OpenAI-compatible server accepts multi-modal data via the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat). Media inputs also support optional UUIDs users can provide to uniquely identify each media, which is used to cache the media results across requests.

!!! important
    A chat template is **required** to use Chat Completions API.
    For HF format models, the default chat template is defined inside `chat_template.json` or `tokenizer_config.json`.

    If no default chat template is available, we will first look for a built-in fallback in [vllm/transformers_utils/chat_templates/registry.py](../../vllm/transformers_utils/chat_templates/registry.py).
    If no fallback is available, an error is raised and you have to provide the chat template manually via the `--chat-template` argument.

    For certain models, we provide alternative chat templates inside [examples](../../examples).
    For example, VLM2Vec uses [examples/template_vlm2vec_phi3v.jinja](../../examples/template_vlm2vec_phi3v.jinja) which is different from the default one for Phi-3-Vision.

### Image Inputs

Image input is supported according to [OpenAI Vision API](https://platform.openai.com/docs/guides/vision).
Here is a simple example using Phi-3.5-Vision.

First, launch the OpenAI-compatible server:

```bash
vllm serve microsoft/Phi-3.5-vision-instruct --runner generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt '{"image":2}'
```

Then, you can use the OpenAI client as follows:

??? code

    ```python
    import os
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Single-image input inference

    # Public image URL for testing remote image processing
    image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # Create chat completion with remote image
    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    # NOTE: The prompt formatting with the image token `<image>` is not needed
                    # since the prompt will be processed automatically by the API server.
                    {
                        "type": "text",
                        "text": "What’s in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_url,  # Optional
                    },
                ],
            }
        ],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)

    # Local image file path (update this to point to your actual image file)
    image_file = "/path/to/image.jpg"

    # Create chat completion with local image file
    # Launch the API server/engine with the --allowed-local-media-path argument.
    if os.path.exists(image_file):
        chat_completion_from_local_image_url = client.chat.completions.create(
            model="microsoft/Phi-3.5-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_file}"},
                        },
                    ],
                }
            ],
        )
        result = chat_completion_from_local_image_url.choices[0].message.content
        print("Chat completion output from local image file:\n", result)
    else:
        print(f"Local image file not found at {image_file}, skipping local file test.")

    # Multi-image input inference
    image_url_duck = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"
    image_url_lion = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg"

    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the animals in these images?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_duck},
                        "uuid": image_url_duck,  # Optional
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_lion},
                        "uuid": image_url_lion,  # Optional
                    },
                ],
            }
        ],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)
    ```

Full example: [examples/online_serving/openai_chat_completion_client_for_multimodal.py](../../examples/online_serving/openai_chat_completion_client_for_multimodal.py)

!!! tip
    Loading from local file paths is also supported on vLLM: You can specify the allowed local media path via `--allowed-local-media-path` when launching the API server/engine,
    and pass the file path as `url` in the API request.

!!! tip
    There is no need to place image placeholders in the text content of the API request - they are already represented by the image content.
    In fact, you can place image placeholders in the middle of the text by interleaving text and image content.

!!! note
    By default, the timeout for fetching images through HTTP URL is `5` seconds.
    You can override this by setting the environment variable:

    ```bash
    export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>
    ```

### Video Inputs

Instead of `image_url`, you can pass a video file via `video_url`. Here is a simple example using [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf).

First, launch the OpenAI-compatible server:

```bash
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --runner generate --max-model-len 8192
```

Then, you can use the OpenAI client as follows:

??? code

    ```python
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"

    ## Use video url in the payload
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this video?",
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "uuid": video_url,  # Optional
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:", result)
    ```

Full example: [examples/online_serving/openai_chat_completion_client_for_multimodal.py](../../examples/online_serving/openai_chat_completion_client_for_multimodal.py)

!!! note
    By default, the timeout for fetching videos through HTTP URL is `30` seconds.
    You can override this by setting the environment variable:

    ```bash
    export VLLM_VIDEO_FETCH_TIMEOUT=<timeout>
    ```

#### Video Frame Recovery

For improved robustness when processing potentially corrupted or truncated video files, vLLM supports optional frame recovery using a dynamic window forward-scan approach. When enabled, if a target frame fails to load during sequential reading, the next successfully grabbed frame (before the next target frame) will be used in its place.

To enable video frame recovery, pass the `frame_recovery` parameter via `--media-io-kwargs`:

```bash
# Example: Enable frame recovery
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"frame_recovery": true}}'
```

**Parameters:**

- `frame_recovery`: Boolean flag to enable forward-scan recovery. When `true`, failed frames are recovered using the next available frame within the dynamic window (up to the next target frame). Default is `false`.

**How it works:**

1. The system reads frames sequentially
2. If a target frame fails to grab, it's marked as "failed"
3. The next successfully grabbed frame (before reaching the next target) is used to recover the failed frame
4. This approach handles both mid-video corruption and end-of-video truncation

Works with common video formats like MP4 when using OpenCV backends.

#### Custom RGBA Background Color

To use a custom background color for RGBA images, pass the `rgba_background_color` parameter via `--media-io-kwargs`:

```bash
# Example: Black background for dark theme
vllm serve llava-hf/llava-1.5-7b-hf \
  --media-io-kwargs '{"image": {"rgba_background_color": [0, 0, 0]}}'

# Example: Custom gray background
vllm serve llava-hf/llava-1.5-7b-hf \
  --media-io-kwargs '{"image": {"rgba_background_color": [128, 128, 128]}}'
```

### Audio Inputs

Audio input is supported according to [OpenAI Audio API](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in).
Here is a simple example using Ultravox-v0.5-1B.

First, launch the OpenAI-compatible server:

```bash
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

Then, you can use the OpenAI client as follows:

??? code

    ```python
    import base64
    import requests
    from openai import OpenAI
    from vllm.assets.audio import AudioAsset

    def encode_base64_content_from_url(content_url: str) -> str:
        """Encode a content retrieved from a remote url to base64 format."""

        with requests.get(content_url) as response:
            response.raise_for_status()
            result = base64.b64encode(response.content).decode('utf-8')

        return result

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Any format supported by librosa is supported
    audio_url = AudioAsset("winning_call").url
    audio_base64 = encode_base64_content_from_url(audio_url)

    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this audio?",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav",
                        },
                        "uuid": audio_url,  # Optional
                    },
                ],
            },
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from input audio:", result)
    ```

Alternatively, you can pass `audio_url`, which is the audio counterpart of `image_url` for image input:

??? code

    ```python
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this audio?",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                        "uuid": audio_url,  # Optional
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from audio url:", result)
    ```

Full example: [examples/online_serving/openai_chat_completion_client_for_multimodal.py](../../examples/online_serving/openai_chat_completion_client_for_multimodal.py)

!!! note
    By default, the timeout for fetching audios through HTTP URL is `10` seconds.
    You can override this by setting the environment variable:

    ```bash
    export VLLM_AUDIO_FETCH_TIMEOUT=<timeout>
    ```

### Embedding Inputs

To input pre-computed embeddings belonging to a data type (i.e. image, video, or audio) directly to the language model,
pass a tensor of shape `(num_items, feature_size, hidden_size of LM)` to the corresponding field of the multi-modal dictionary.

You must enable this feature via the `--enable-mm-embeds` flag in `vllm serve`.

!!! warning
    The vLLM engine may crash if incorrect shape of embeddings is passed.
    Only enable this flag for trusted users!

#### Image Embedding Inputs

For image embeddings, you can pass the base64-encoded tensor to the `image_embeds` field.
The following example demonstrates how to pass image embeddings to the OpenAI server:

??? code

    ```python
    from vllm.utils.serial_utils import tensor2base64

    image_embedding = torch.load(...)
    grid_thw = torch.load(...) # Required by Qwen/Qwen2-VL-2B-Instruct

    base64_image_embedding = tensor2base64(image_embedding)

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Basic usage - this is equivalent to the LLaVA example for offline inference
    model = "llava-hf/llava-1.5-7b-hf"
    embeds = {
        "type": "image_embeds",
        "image_embeds": f"{base64_image_embedding}",
        "uuid": image_url,  # Optional
    }

    # Pass additional parameters (available to Qwen2-VL and MiniCPM-V)
    model = "Qwen/Qwen2-VL-2B-Instruct"
    embeds = {
        "type": "image_embeds",
        "image_embeds": {
            "image_embeds": f"{base64_image_embedding}",  # Required
            "image_grid_thw": f"{base64_image_grid_thw}",  # Required by Qwen/Qwen2-VL-2B-Instruct
        },
        "uuid": image_url,  # Optional
    }
    model = "openbmb/MiniCPM-V-2_6"
    embeds = {
        "type": "image_embeds",
        "image_embeds": {
            "image_embeds": f"{base64_image_embedding}",  # Required
            "image_sizes": f"{base64_image_sizes}",  # Required by openbmb/MiniCPM-V-2_6
        },
        "uuid": image_url,  # Optional
    }
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    embeds,
                ],
            },
        ],
        model=model,
    )
    ```

For Online Serving, you can also skip sending media if you expect cache hits with provided UUIDs. You can do so by sending media like this:

??? code

    ```python
        # Image/video/audio URL:
        {
            "type": "image_url",
            "image_url": None,
            "uuid": image_uuid,
        },

        # image_embeds
        {
            "type": "image_embeds",
            "image_embeds": None,
            "uuid": image_uuid,
        },

        # input_audio:
        {
            "type": "input_audio",
            "input_audio": None,
            "uuid": audio_uuid,
        },

        # PIL Image:
        {
            "type": "image_pil",
            "image_pil": None,
            "uuid": image_uuid,
        },

    ```

!!! note
    Multiple messages can now contain `{"type": "image_embeds"}`, enabling you to pass multiple image embeddings in a single request (similar to regular images). The number of embeddings is limited by `--limit-mm-per-prompt`.

    **Important**: The embedding shape format differs based on the number of embeddings:

    - **Single embedding**: 3D tensor of shape `(1, feature_size, hidden_size)`
    - **Multiple embeddings**: List of 2D tensors, each of shape `(feature_size, hidden_size)`

    If used with a model that requires additional parameters, you must also provide a tensor for each of them, e.g. `image_grid_thw`, `image_sizes`, etc.
