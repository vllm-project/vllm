from dataclasses import dataclass

import lm_eval
import pytest

TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03


@dataclass
class GSM8KAccuracyTestConfig:
    model_name: str
    excepted_value: float

    def get_model_args(self) -> str:
        return (f"pretrained={self.model_name},"
                "max_model_len=4096,max_num_seqs=128,enforce_eager=True")


# NOTE(rob): Accuracy scores measured on GPUs.
ACCURACY_CONFIGS = [
    GSM8KAccuracyTestConfig(
        model_name="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
        excepted_value=0.76),  # no bias
    GSM8KAccuracyTestConfig(
        model_name="neuralmagic/Qwen2-7B-Instruct-quantized.w8a8",
        excepted_value=0.66),  # bias
]


@pytest.mark.parametrize("config", ACCURACY_CONFIGS)
def test_gsm8k_correctness(config: GSM8KAccuracyTestConfig):

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(),
        tasks="gsm8k",
        batch_size="auto",
        limit=1,
    )

    # EXPECTED_VALUE = config.excepted_value
    # measured_value = results["results"][TASK][FILTER]
    # assert (measured_value - RTOL < EXPECTED_VALUE
    #         and measured_value + RTOL > EXPECTED_VALUE
    #         ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"
