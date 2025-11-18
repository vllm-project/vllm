"""Lab 08: Multi-Model Serving - Complete Solution"""

from typing import Dict, Optional
from vllm import LLM, SamplingParams


class ModelRegistry:
    """Registry for managing multiple models."""

    def __init__(self):
        self.models: Dict[str, LLM] = {}

    def load_model(self, name: str, model_path: str) -> None:
        """Load a model into the registry."""
        self.models[name] = LLM(model=model_path, trust_remote_code=True)
        print(f"Loaded model: {name}")

    def get_model(self, name: str) -> Optional[LLM]:
        """Get a model by name."""
        return self.models.get(name)

    def route_request(self, model_name: str, prompt: str) -> str:
        """Route request to specified model."""
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
        outputs = model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text


def main():
    """Main multi-model demo."""
    print("=== Multi-Model Serving Lab ===\n")

    registry = ModelRegistry()

    # Load models
    print("Loading models...")
    registry.load_model("opt", "facebook/opt-125m")
    registry.load_model("gpt2", "gpt2")

    # Route requests
    print("\nRouting requests...")
    result1 = registry.route_request("opt", "Hello")
    print(f"OPT: {result1[:50]}...")

    result2 = registry.route_request("gpt2", "Hello")
    print(f"GPT2: {result2[:50]}...")


if __name__ == "__main__":
    main()
