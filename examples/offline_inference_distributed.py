from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray

ray.init(
    runtime_env={
        "pip": [
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "numpy<1.24",  # remove when mlflow updates beyond 2.2
            "torch",
        ]
    }
)

import ray.data

ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model="facebook/opt-125m")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        return {"output": self.llm.generate(batch["text"], sampling_params)}

scale = ray.data.ActorPoolStrategy(size=10)
predictions = ds.map_batches(LLMPredictor, compute=scale, num_gpus=1, batch_size=32)
predictions.show(limit=1)
