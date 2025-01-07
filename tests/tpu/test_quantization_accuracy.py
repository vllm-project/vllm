import lm_eval

MODEL_NAME="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
MODEL_ARGS = f"pretrained={MODEL_NAME},max_model_len=4096,max_num_seqs=128"
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
EXPECTED_VALUE = 0.58

def test_w8a8_single():
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=MODEL_NAME,
        tasks="gsm8k",
        batch_size="auto",
    )
    print(results)

    measured_value = results["results"][TASK][FILTER]
    assert (measured_value - RTOL < EXPECTED_VALUE
            and measured_value + RTOL > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"
