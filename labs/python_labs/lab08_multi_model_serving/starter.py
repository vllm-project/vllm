"""Lab 08: Multi-Model Serving - Starter Code"""

from typing import Dict, Optional
from vllm import LLM, SamplingParams


class ModelRegistry:
    """Registry for managing multiple models."""

    def __init__(self):
        self.models: Dict[str, LLM] = {}

    def load_model(self, name: str, model_path: str) -> None:
        """Load a model into the registry."""
        # TODO 1: Load and register model
        pass

    def get_model(self, name: str) -> Optional[LLM]:
        """Get a model by name."""
        # TODO 2: Retrieve model from registry
        pass

    def route_request(self, model_name: str, prompt: str) -> str:
        """Route request to specified model."""
        # TODO 3: Route and generate
        pass


def main():
    """Main multi-model demo."""
    print("=== Multi-Model Serving Lab ===\n")

    # TODO 4: Load multiple models
    # TODO 5: Route requests to different models


if __name__ == "__main__":
    main()
