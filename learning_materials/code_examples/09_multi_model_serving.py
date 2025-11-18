"""
Example 09: Multi-Model Serving

Demonstrates serving multiple models and routing requests.

Usage:
    python 09_multi_model_serving.py
"""

from typing import Dict
from vllm import LLM, SamplingParams


class ModelServer:
    """Serve multiple models with routing."""

    def __init__(self):
        self.models: Dict[str, LLM] = {}

    def add_model(self, name: str, model_path: str) -> None:
        """Add a model to the server."""
        print(f"Loading {name}...")
        self.models[name] = LLM(model=model_path, trust_remote_code=True)
        print(f"  {name} loaded successfully")

    def generate(self, model_name: str, prompt: str, max_tokens: int = 50) -> str:
        """Generate using specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        sampling_params = SamplingParams(temperature=0.8, max_tokens=max_tokens)
        output = model.generate([prompt], sampling_params)[0]
        return output.outputs[0].text


def main():
    """Demo multi-model serving."""
    print("=== Multi-Model Serving Demo ===\n")

    server = ModelServer()

    # Load models
    server.add_model("small", "facebook/opt-125m")
    # server.add_model("medium", "facebook/opt-350m")  # Uncomment if you have memory

    print("\nGenerating with different models:\n")

    # Generate with small model
    result = server.generate("small", "The future of AI", max_tokens=30)
    print(f"Small model: {result}")


if __name__ == "__main__":
    main()
