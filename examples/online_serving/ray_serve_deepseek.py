# SPDX-License-Identifier: Apache-2.0
"""
Example to deploy DeepSeek R1 or V3 with Ray Serve LLM.
See Ray Serve LLM documentation at:
https://docs.ray.io/en/latest/serve/llm/serving-llms.html

Run `python3 ray_serve_deepseek.py` to deploy the model.
"""

from ray import serve
from ray.serve.llm import LLMConfig, LLMRouter, LLMServer

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="deepseek",
        # Change to model download path
        model_source="/path/to/the/model",
    ),
    deployment_config=dict(autoscaling_config=dict(
        min_replicas=1,
        max_replicas=1,
    )),
    # Change to the accelerator type of the node
    accelerator_type="H100",
    runtime_env=dict(env_vars=dict(VLLM_USE_V1="1")),
    # Customize engine arguments as needed (e.g. vLLM engine kwargs)
    engine_kwargs=dict(
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.92,
        dtype="auto",
        max_num_seqs=40,
        max_model_len=16384,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
    ),
)

# Deploy the application
deployment = LLMServer.as_deployment(
    llm_config.get_serve_options(name_prefix="vLLM:")).bind(llm_config)
llm_app = LLMRouter.as_deployment().bind([deployment])
serve.run(llm_app)
