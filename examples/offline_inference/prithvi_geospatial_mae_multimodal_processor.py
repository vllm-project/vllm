# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import LLMForImageTiling

# install plugin at: https://github.com/christian-pinto/prithvi_multimodal_processor_plugin

torch.set_default_dtype(torch.float16)

image_url = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/main/examples/India_900498_S2Hand.tif"  # noqa: E501

llm = LLMForImageTiling(
    model="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
    skip_tokenizer_init=True,
    trust_remote_code=True,
    enforce_eager=True,
    # Limit the maximum number of parallel requests
    # to avoid the model going OOM
    max_num_seqs=32,
)

prompt = {"type": "url", "format": "geotiff", "data": image_url}

output = llm.predict(prompt)
print(output)
print(output.data)
