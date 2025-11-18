"""
Example 15: Logging Configuration

Demonstrates proper logging setup for production applications.

Usage:
    python 15_logging_configuration.py
"""

import logging
import sys
from vllm import LLM, SamplingParams


def setup_logging(level=logging.INFO):
    """Configure structured logging."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler('vllm_app.log')
    file_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def main():
    """Demo logging configuration."""
    # Setup logging
    logger = setup_logging(logging.INFO)

    logger.info("=== Logging Demo Started ===")

    try:
        logger.info("Initializing vLLM model...")
        llm = LLM(model="facebook/opt-125m", trust_remote_code=True)
        logger.info("Model loaded successfully")

        logger.info("Preparing inference request...")
        sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
        prompt = "Machine learning is"

        logger.info(f"Generating completion for prompt: '{prompt}'")
        output = llm.generate([prompt], sampling_params)[0]

        logger.info(f"Generation complete. Output length: {len(output.outputs[0].text)} chars")
        logger.debug(f"Full output: {output.outputs[0].text}")

        print(f"\nGenerated: {output.outputs[0].text}")

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise

    logger.info("=== Logging Demo Complete ===")
    print("\nCheck vllm_app.log for detailed logs")


if __name__ == "__main__":
    main()
