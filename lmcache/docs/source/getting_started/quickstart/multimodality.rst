Example: Multimodal KV Cache Support
====================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


Quick Start Example (Audio Model): 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We  going to be running audio inference with ``ultravox-v0_5-llama-3_2-1b`` and using LMCache to speed up the TTFT after the first request.

**Install and Serve:** 

``pip install lmcache vllm[audio] openai``

.. code-block:: bash

   vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
       --max-model-len 4096 --trust-remote-code \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'


**Benchmark:** 

Save as ``audio_query.py``

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   # SPDX-FileCopyrightText: Copyright contributors to the vLLM project

   import base64

   import requests
   from openai import OpenAI

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
       
   start_time = time.time()

   model = get_first_model(client)
   run_audio(model)
   end_time = time.time()
   print(f"Time taken: {end_time - start_time} seconds")   


**Run and see TTFT speedup:** 

.. code-block:: bash

   # first time: 
   python audio_query.py

   # second time: 
   python audio_query.py


**Retrieval and speed up in logs:**

1. After First Request:

.. code-block:: text

   [2025-08-05 09:58:06,965] LMCache INFO: Reqid: chatcmpl-dd6e8a131f2b455fa3cd133a9bfab26f, Total tokens 201, LMCache hit tokens: 201, need to load: 8 (vllm_v1_adapter.py:803:lmcache.integration.vllm.vllm_v1_adapter)
   [2025-08-05 09:58:06,967] LMCache INFO: Retrieved 201 out of 201 out of total 201 tokens (cache_engine.py:500:lmcache.v1.cache_engine)
   [2025-08-05 09:58:07,178] LMCache INFO: Storing KV cache for 256 out of 256 tokens (skip_leading_tokens=0) for request chatcmpl-dd6e8a131f2b455fa3cd133a9bfab26f (vllm_v1_adapter.py:709:lmcache.integration.vllm.vllm_v1_adapter)
   [2025-08-05 09:58:07,178] LMCache INFO: Stored 256 out of total 256 tokens. size: 0.0078 gb, cost 0.5096 ms, throughput: 15.3291 GB/s; offload_time: 0.4897 ms, put_time: 0.0200 ms (cache_engine.py:251:lmcache.v1.cache_engine)

*Example Output:*

.. code-block:: text

   Chat completion output from input audio: It seems like you're excitedly sharing your thoughts and predictions about a game you're about to watch. The audio appears to be a stream of text messages or social media updates. The words and phrases you've copied seem to indicate that you're a sports fan, particularly in Major League Baseball (MLB). 

   Are
   Chat completion output from audio url: It appears to be a enthusiastic and excited baseball comment from an individual. The language used, such as "And the one pitch on the way to Edgar Martinez has swung on and line down the line for a base hit," suggests a strong amateur athlete's excitement and commentary. The reference to the playoff qualification and the praise for
   Chat completion output from base64 encoded audio: It seems like you're excited about a sports game, possibly the California Athletics (now known as the Los Angeles Angels), given the reference to Edgar Martinez and the Birds (no team by that name in the AL) in the mixed messages.

   However, I'm not seeing any audio in the conversation. Are you referring to
   Time taken: 37.96290421485901 seconds

2. After Second Request: 

.. code-block:: text

   [2025-08-05 09:58:07,371] LMCache INFO: Reqid: chatcmpl-2a130545a6a24f33b41e219ef0807a61, Total tokens 201, LMCache hit tokens: 201, need to load: 8 (vllm_v1_adapter.py:803:lmcache.integration.vllm.vllm_v1_adapter)
   [2025-08-05 09:58:07,372] LMCache INFO: Retrieved 201 out of 201 out of total 201 tokens (cache_engine.py:500:lmcache.v1.cache_engine)
   [2025-08-05 09:58:07,558] LMCache INFO: Storing KV cache for 256 out of 256 tokens (skip_leading_tokens=0) for request chatcmpl-2a130545a6a24f33b41e219ef0807a61 (vllm_v1_adapter.py:709:lmcache.integration.vllm.vllm_v1_adapter)
   [2025-08-05 09:58:07,558] LMCache INFO: Stored 256 out of total 256 tokens. size: 0.0078 gb, cost 0.4962 ms, throughput: 15.7450 GB/s; offload_time: 0.4782 ms, put_time: 0.0179 ms (cache_engine.py:251:lmcache.v1.cache_engine)

*Example Output:*

.. code-block:: text

   Chat completion output from input audio: It seems like you're extremely excited about the possibility of the San Francisco Giants winning the American League championship and playing in the World Series. The audio is filled with emotions and a sense of optimism, with you enthusiastically expressing your thoughts and feelings. It's clear that this is a significant moment for you, particularly given the fact
   Chat completion output from audio url: I can tell you're excited about a baseball game. It seems like you're reliving a moment during the middle of a game, especially the highlight of a six runs game for the Golden Giants. The audio appears to include a local sports radio talk show style broadcast, with a narrator (or DJs) discussing the importance
   Chat completion output from base64 encoded audio: It seems like you're having a lively discussion about baseball, specifically about the Arizona Diamondbacks and their chances of winning the American League championship. You're using colloquial expressions and slang, such as "the Oone hitter," " rejoice," and "waving him in." These cues suggest that you're engaged in
   Time taken: 5.39893364906311 seconds