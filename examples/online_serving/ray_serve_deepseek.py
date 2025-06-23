# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example to deploy DeepSeek R1 or V3 with Ray Serve LLM.
See more details at:
https://docs.ray.io/en/latest/serve/tutorials/serve-deepseek.html
And see Ray Serve LLM documentation at:
https://docs.ray.io/en/latest/serve/llm/serving-llms.html

Run `python3 ray_serve_deepseek.py` to deploy the model.
"""

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "deepseek",
        # Since DeepSeek model is huge, it is recommended to pre-download
        # the model to local disk, say /path/to/the/model and specify:
        # model_source="/path/to/the/model"
        "model_source": "deepseek-ai/DeepSeek-R1",
    },
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 1,
        }
    },
    # Change to the accelerator type of the node
    accelerator_type="H100",
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
    # Customize engine arguments as needed (e.g. vLLM engine kwargs)
    engine_kwargs={
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 2,
        "gpu_memory_utilization": 0.92,
        "dtype": "auto",
        "max_num_seqs": 40,
        "max_model_len": 16384,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
    },
)

# Deploy the application
llm_app = build_openai_app({"llm_configs": [llm_config]})
serve.run(llm_app)
