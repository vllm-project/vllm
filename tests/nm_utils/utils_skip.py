"""Checks environment variables to skip various test groups.
The functions here are imported by each test file.
The .github/actions/nm-test-skipping-env-setup sets these
    variables in the testing automation.
"""

import os


def should_skip_accuracy_test_group():
    TEST_ACCURACY = os.getenv("TEST_ACCURACY", "ENABLE")
    return TEST_ACCURACY == "DISABLE"


def should_skip_async_engine_test_group():
    TEST_ASYNC_ENGINE = os.getenv("TEST_ASYNC_ENGINE", "ENABLE")
    return TEST_ASYNC_ENGINE == "DISABLE"


def should_skip_basic_correctness_test_group():
    TEST_BASIC_CORRECTNESS = os.getenv("TEST_BASIC_CORRECTNESS", "ENABLE")
    return TEST_BASIC_CORRECTNESS == "DISABLE"


def should_skip_core_test_group():
    TEST_CORE = os.getenv("TEST_CORE", "ENABLE")
    return TEST_CORE == "DISABLE"


def should_skip_distributed_test_group():
    TEST_DISTRIBUTED = os.getenv("TEST_DISTRIBUTED", "ENABLE")
    return TEST_DISTRIBUTED == "DISABLE"


def should_skip_engine_test_group():
    TEST_ENGINE = os.getenv("TEST_ENGINE", "ENABLE")
    return TEST_ENGINE == "DISABLE"


def should_skip_entrypoints_test_group():
    TEST_ENTRYPOINTS = os.getenv("TEST_ENTRYPOINTS", "ENABLE")
    return TEST_ENTRYPOINTS == "DISABLE"


def should_skip_kernels_test_groups():
    TEST_KERNELS = os.getenv("TEST_KERNELS", "ENABLE")
    return TEST_KERNELS == "DISABLE"


def should_skip_lora_test_group():
    TEST_LORA = os.getenv("TEST_LORA", "ENABLE")
    return TEST_LORA == "DISABLE"


def should_skip_metrics_test_group():
    TEST_METRICS = os.getenv("TEST_METRICS", "ENABLE")
    return TEST_METRICS == "DISABLE"


def should_skip_model_executor_test_group():
    TEST_MODEL_EXECUTOR = os.getenv("TEST_MODEL_EXECUTOR", "ENABLE")
    return TEST_MODEL_EXECUTOR == "DISABLE"


def should_skip_models_test_group():
    TEST_MODELS = os.getenv("TEST_MODELS", "ENABLE")
    return TEST_MODELS == "DISABLE"


def should_skip_models_core_test_group():
    TEST_MODELS_CORE = os.getenv("TEST_MODELS_CORE", "ENABLE")
    return TEST_MODELS_CORE == "DISABLE"


def should_skip_prefix_caching_test_group():
    TEST_PREFIX_CACHING = os.getenv("TEST_PREFIX_CACHING", "ENABLE")
    return TEST_PREFIX_CACHING == "DISABLE"


def should_skip_quantization_test_group():
    TEST_QUANTIZATION = os.getenv("TEST_QUANTIZATION", "ENABLE")
    return TEST_QUANTIZATION == "DISABLE"


def should_skip_samplers_test_group():
    TEST_SAMPLERS = os.getenv("TEST_SAMPLERS", "ENABLE")
    return TEST_SAMPLERS == "DISABLE"


def should_skip_spec_decode_test_group():
    TEST_SPEC_DECODE = os.getenv("TEST_SPEC_DECODE", "ENABLE")
    return TEST_SPEC_DECODE == "DISABLE"


def should_skip_tensorizer_loader_test_group():
    TEST_TENSORIZER_LOADER = os.getenv("TEST_TENSORIZER_LOADER", "ENABLE")
    return TEST_TENSORIZER_LOADER == "DISABLE"


def should_skip_tokenization_test_group():
    TEST_TOKENIZATION = os.getenv("TEST_TOKENIZATION", "ENABLE")
    return TEST_TOKENIZATION == "DISABLE"


def should_skip_worker_test_group():
    TEST_WORKER = os.getenv("TEST_WORKER", "ENABLE")
    return TEST_WORKER == "DISABLE"


MAP = {
    "TEST_ACCURACY": should_skip_accuracy_test_group,
    "TEST_ASYNC_ENGINE": should_skip_async_engine_test_group,
    "TEST_BASIC_CORRECTNESS": should_skip_basic_correctness_test_group,
    "TEST_CORE": should_skip_core_test_group,
    "TEST_DISTRIBUTED": should_skip_distributed_test_group,
    "TEST_ENGINE": should_skip_engine_test_group,
    "TEST_ENTRYPOINTS": should_skip_entrypoints_test_group,
    "TEST_KERNELS": should_skip_kernels_test_groups,
    "TEST_LORA": should_skip_lora_test_group,
    "TEST_METRICS": should_skip_metrics_test_group,
    "TEST_MODELS": should_skip_models_test_group,
    "TEST_MODELS_CORE": should_skip_models_core_test_group,
    "TEST_PREFIX_CACHING": should_skip_prefix_caching_test_group,
    "TEST_QUANTIZATION": should_skip_quantization_test_group,
    "TEST_SAMPLERS": should_skip_samplers_test_group,
    "TEST_SPEC_DECODE": should_skip_spec_decode_test_group,
    "TEST_TENSORIZER_LOADER": should_skip_tensorizer_loader_test_group,
    "TEST_TOKENIZATION": should_skip_tokenization_test_group,
    "TEST_WORKER": should_skip_worker_test_group,
}


def should_skip_test_group(group_name: str) -> bool:
    return MAP[group_name]()
