"""
Example 16: Request Validation with Pydantic

Shows input validation using Pydantic models.

Usage:
    python 16_request_validation.py
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from vllm import LLM, SamplingParams


class InferenceRequest(BaseModel):
    """Validated inference request."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop_sequences: Optional[list[str]] = None

    @validator('prompt')
    def validate_prompt(cls, v):
        """Custom prompt validation."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v.strip()

    @validator('temperature')
    def validate_temperature(cls, v, values):
        """Ensure temperature makes sense."""
        if v == 0.0 and 'top_p' in values:
            # Greedy decoding, top_p is ignored
            pass
        return v


def generate_with_validation(request: InferenceRequest) -> str:
    """Generate text with validated input."""
    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop_sequences
    )

    output = llm.generate([request.prompt], sampling_params)[0]
    return output.outputs[0].text


def main():
    """Demo request validation."""
    print("=== Request Validation Demo ===\n")

    # Valid request
    try:
        request = InferenceRequest(
            prompt="The future of AI",
            max_tokens=50,
            temperature=0.8
        )
        print(f"Valid request: {request.prompt}")
        result = generate_with_validation(request)
        print(f"Result: {result[:50]}...\n")
    except Exception as e:
        print(f"Validation error: {e}\n")

    # Invalid request (empty prompt)
    try:
        request = InferenceRequest(prompt="   ")
    except Exception as e:
        print(f"Caught validation error: {e}\n")

    # Invalid request (temperature out of range)
    try:
        request = InferenceRequest(
            prompt="Test",
            temperature=3.0
        )
    except Exception as e:
        print(f"Caught validation error: {e}\n")


if __name__ == "__main__":
    main()
