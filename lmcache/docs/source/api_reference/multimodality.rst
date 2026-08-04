KV Caching for Multimodal Models with vLLM
##########################################

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


Overview
********

vLLM is building on its multimodal capability and currently supports the following `List of Multimodal Language Models <https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-multimodal-language-models>`_. 

LMCache can therefore be used to speed up inference time for all multimodal models supported by vLLM. This document shows the speedup improvements using LMCache for KV caching in vLLM for multimodal models.

Examples of TTFT speed up for different multimodal types 
========================================================

Prerequisites
-------------

- A Machine with at least one GPU. You can adjust the max model length of your vLLM instance depending on your GPU memory
- vLLM and LMCache installed (:doc:`Installation Guide <../getting_started/installation>`)
- vLLM audio dependencies installed: ``pip install vllm[audio]``

Examples
--------

.. note::

    The examples below use a python script for inferencing multimodal models hosted by vLLM.
    The script is the `openai_chat_completion_client_for_multimodal python script in vLLM <https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py>`_.
    You will need to download it locally for running the examples below.
    The script is printed in the `reference section <#reference-inferencing-multimodal-models-in-vllm-example-python-script>`_ that follows for you perusal.
    Go to the `Example output <#example-output>`_ section to see the output in the vLLM logs that demonstrate the speedup improvements.


Audio Inference with Ultravox:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start vLLM server with ``fixie-ai/ultravox-v0_5-llama-3_2-1b`` model and LMCache KV caching:

.. code-block:: bash

   vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
       --max-model-len 4096 --trust-remote-code \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Run the python script twice to demonstrate TTFT speedup on the second turn because of the caching:

.. code-block:: bash

   # run twice to see TTFT speedup
   python openai_chat_completion_client_for_multimodal.py --chat-type audio

Single Image Inference with Llava:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start vLLM server with ``llava-hf/llava-1.5-7b-hf`` model and LMCache KV caching:

.. code-block:: bash

   vllm serve llava-hf/llava-1.5-7b-hf \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Run the python script twice to demonstrate TTFT speedup on the second turn because of the caching:

.. code-block:: bash

   # run twice to see TTFT speedup
   python openai_chat_completion_client_for_multimodal.py --chat-type single-image

Multi-image Inference with Phi-3.5-vision-instruct:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start vLLM server with ``microsoft/Phi-3.5-vision-instruct`` model and LMCache KV caching:

.. code-block:: bash

   vllm serve microsoft/Phi-3.5-vision-instruct \
       --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt '{"image":2}' \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Run the python script twice to demonstrate TTFT speedup on the second turn because of the caching:

.. code-block:: bash

   # run twice to see TTFT speedup
   python openai_chat_completion_client_for_multimodal.py --chat-type multi-image

Video Inference with Llava-OneVision:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start vLLM server with ``llava-hf/llava-onevision-qwen2-7b-ov-hf`` model and LMCache KV caching:

.. code-block:: bash

   vllm serve llava-hf/llava-onevision-qwen2-7b-ov-hf \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

Run the python script twice to demonstrate TTFT speedup on the second turn because of the caching:

.. code-block:: bash

   # run twice to see TTFT speedup
   python openai_chat_completion_client_for_multimodal.py --chat-type video


Example output
--------------

When running the examples above you will notice output in the vLLM logs similar to below. 

This first output demonstrates the tokens being cached and loaded.

.. code-block:: text

   [2025-08-04 22:43:35,484] LMCache INFO: Reqid: chatcmpl-05e2d296601046b29210f53a1fa30b13, Total tokens 1536, LMCache hit tokens: 1536, need to load: 1535 (vllm_v1_adapter.py:803:lmcache.integration.vllm.vllm_v1_adapter)

This then shows the speedup between the first and second runs.

1. First request: 

.. code-block:: text

   Chat completion output from input audio: It seems like you're excitedly sharing your thoughts and predictions about a game you're about to watch. The audio appears to be a stream of comments or a social media post. The words "one pitch on the way to Edgar Martinez" suggest that someone is saying something in a baseball chat or social media post about the
   Chat completion output from audio url: It appears to be a enthusiastic and excited baseball comment from an individual. The content seems to be a play-by-play description of a specific baseball game, with the narrator belonging to a team that is competing in the American League Championship Series. The reference to the player Edgar Martinez is a nod to a well-known baseball player,
   Chat completion output from base64 encoded audio: It seems like you're excited about a baseball game, possibly the Los Angeles Dodgers or the Boston Red Sox, but it's unclear which one. The text mentions a "pitcher" and "swung on the line," but it's not entirely obvious which team it's referring to.
   
   However, the mention of "
   Time taken: 50.828808307647705 seconds

2. Second request: 

.. code-block:: text

   Chat completion output from input audio: It seems like you're extremely excited about the possibility of the San Francisco Giants winning the American League championship and playing in the World Series. The audio is filled with emotions and a sense of optimism, with you enthusiastically expressing your thoughts and feelings. It's clear that this is a significant moment for you, particularly given the fact
   Chat completion output from audio url: I can tell you're excited about a baseball game. It seems like you're reliving a moment during the middle of a game, especially the highlight of a six runs game for the Golden Giants. The audio appears to include a local sports radio talk show style broadcast, with a ringer ("the guy" in the
   Chat completion output from base64 encoded audio: It seems like you're having a lively discussion about a Major League Baseball game, specifically about the shortstop playing for the Mariners and,Mario Upton swinging at a pitch and eventually being thrown out on a play at the plate. The atmosphere is excited, with all the cheering and commentary you've written. It appears to
   Time taken: 3.3407371044158936 seconds


Reference: Inferencing multimodal models in vLLM example Python script 
======================================================================

Source: https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
   
   import base64
   
   import requests
   from openai import OpenAI
   
   from vllm.utils import FlexibleArgumentParser
   
   # SPDX-License-Identifier: Apache-2.0
   # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
   from openai import APIConnectionError, OpenAI
   from openai.pagination import SyncPage
   from openai.types.model import Model
   
   import time
   
   
   def get_first_model(client: OpenAI) -> str:
       """
       Get the first model from the vLLM server.
       """
       try:
           models: SyncPage[Model] = client.models.list()
       except APIConnectionError as e:
           raise RuntimeError(
               "Failed to get the list of models from the vLLM server at "
               f"{client.base_url} with API key {client.api_key}. Check\n"
               "1. the server is running\n"
               "2. the server URL is correct\n"
               "3. the API key is correct"
           ) from e
   
       if len(models.data) == 0:
           raise RuntimeError(f"No models found on the vLLM server at {client.base_url}")
   
       return models.data[0].id
   
   # Modify OpenAI's API key and API base to use vLLM's API server.
   openai_api_key = "EMPTY"
   openai_api_base = "http://localhost:8000/v1"
   
   client = OpenAI(
       # defaults to os.environ.get("OPENAI_API_KEY")
       api_key=openai_api_key,
       base_url=openai_api_base,
   )
   
   
   def encode_base64_content_from_url(content_url: str) -> str:
       """Encode a content retrieved from a remote url to base64 format."""
   
       with requests.get(content_url) as response:
           response.raise_for_status()
           result = base64.b64encode(response.content).decode("utf-8")
   
       return result
   
   
   # Text-only inference
   def run_text_only(model: str) -> None:
       chat_completion = client.chat.completions.create(
           messages=[{"role": "user", "content": "What's the capital of France?"}],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion.choices[0].message.content
       print("Chat completion output:", result)
   
   
   # Single-image input inference
   def run_single_image(model: str) -> None:
       ## Use image url in the payload
       image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
       chat_completion_from_url = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What's in this image?"},
                       {
                           "type": "image_url",
                           "image_url": {"url": image_url},
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_url.choices[0].message.content
       print("Chat completion output from image url:", result)
   
       ## Use base64 encoded image in the payload
       image_base64 = encode_base64_content_from_url(image_url)
       chat_completion_from_base64 = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What's in this image?"},
                       {
                           "type": "image_url",
                           "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_base64.choices[0].message.content
       print("Chat completion output from base64 encoded image:", result)
   
   
   # Multi-image input inference
   def run_multi_image(model: str) -> None:
       image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
       image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"
       chat_completion_from_url = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What are the animals in these images?"},
                       {
                           "type": "image_url",
                           "image_url": {"url": image_url_duck},
                       },
                       {
                           "type": "image_url",
                           "image_url": {"url": image_url_lion},
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_url.choices[0].message.content
       print("Chat completion output:", result)
   
   
   # Video input inference
   def run_video(model: str) -> None:
       video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
       video_base64 = encode_base64_content_from_url(video_url)
   
       ## Use video url in the payload
       chat_completion_from_url = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What's in this video?"},
                       {
                           "type": "video_url",
                           "video_url": {"url": video_url},
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_url.choices[0].message.content
       print("Chat completion output from image url:", result)
   
       ## Use base64 encoded video in the payload
       chat_completion_from_base64 = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What's in this video?"},
                       {
                           "type": "video_url",
                           "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_base64.choices[0].message.content
       print("Chat completion output from base64 encoded image:", result)
   
   
   # Audio input inference
   def run_audio(model: str) -> None:
       from vllm.assets.audio import AudioAsset
   
       audio_url = AudioAsset("winning_call").url
       audio_base64 = encode_base64_content_from_url(audio_url)
   
       # OpenAI-compatible schema (`input_audio`)
       chat_completion_from_base64 = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What's in this audio?"},
                       {
                           "type": "input_audio",
                           "input_audio": {
                               # Any format supported by librosa is supported
                               "data": audio_base64,
                               "format": "wav",
                           },
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_base64.choices[0].message.content
       print("Chat completion output from input audio:", result)
   
       # HTTP URL
       chat_completion_from_url = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What's in this audio?"},
                       {
                           "type": "audio_url",
                           "audio_url": {
                               # Any format supported by librosa is supported
                               "url": audio_url
                           },
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_url.choices[0].message.content
       print("Chat completion output from audio url:", result)
   
       # base64 URL
       chat_completion_from_base64 = client.chat.completions.create(
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": "What's in this audio?"},
                       {
                           "type": "audio_url",
                           "audio_url": {
                               # Any format supported by librosa is supported
                               "url": f"data:audio/ogg;base64,{audio_base64}"
                           },
                       },
                   ],
               }
           ],
           model=model,
           max_completion_tokens=64,
       )
   
       result = chat_completion_from_base64.choices[0].message.content
       print("Chat completion output from base64 encoded audio:", result)
   
   
   example_function_map = {
       "text-only": run_text_only,
       "single-image": run_single_image,
       "multi-image": run_multi_image,
       "video": run_video,
       "audio": run_audio,
   }
   
   
   def parse_args():
       parser = FlexibleArgumentParser(
           description="Demo on using OpenAI client for online serving with "
           "multimodal language models served with vLLM."
       )
       parser.add_argument(
           "--chat-type",
           "-c",
           type=str,
           default="single-image",
           choices=list(example_function_map.keys()),
           help="Conversation type with multimodal data.",
       )
       return parser.parse_args()
   
   
   def main(args) -> None:
       chat_type = args.chat_type
       model = get_first_model(client)
       example_function_map[chat_type](model)
   
   
   if __name__ == "__main__":
       args = parse_args()
       start_time = time.time()
       main(args)
       end_time = time.time()
       print(f"Time taken: {end_time - start_time} seconds")

