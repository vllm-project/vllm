#!/bin/bash
# partially copied from from .buildkite/test-pipeline.yml

cd tests || exit 1

# we will need to download test models off HF hub
unset HF_HUB_OFFLINE

# basic correctness
pytest -v -s test_regression.py
pytest -v -s async_engine
VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s basic_correctness/test_basic_correctness.py
VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest -v -s basic_correctness/test_basic_correctness.py
VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s basic_correctness/test_chunked_prefill.py
VLLM_ATTENTION_BACKEND=FLASH_ATTN pytest -v -s basic_correctness/test_chunked_prefill.py
VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 pytest -v -s basic_correctness/test_preemption.py

# core
pytest -v -s core

# note: distributed tests are disabled

# engine tests
pytest -v -s engine tokenization test_sequence.py test_config.py test_logger.py
# entrypoint
pytest -v -s entrypoints -m openai

#inputs (note: multimodal tests are skipped)
pytest -v -s test_inputs.py

#models
pytest -v -s models -m \"not vlm\"

# misc
pytest -v -s prefix_caching
pytest -v -s samplers
pytest -v -s test_logits_processor.py
pytest -v -s models -m \"not vlm\"
pytest -v -s worker
VLLM_ATTENTION_BACKEND=XFORMERS pytest -v -s spec_decode
# pytest -v -s tensorizer_loader # disabled: requires libsodium
pytest -v -s metrics
pytest -v -s quantization
