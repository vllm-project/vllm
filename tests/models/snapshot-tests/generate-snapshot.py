import json
import os

import pytest

from test_models_snapshot import MODELS, DTYPE, MAX_TOKENS

snapshot_path = "./snapshot.json"


def update_snapshot(model, dtype, max_tokens, prompt, hf_output_str):
    key = f"{model}-{dtype}-{max_tokens}"

    if not os.path.exists(snapshot_path):
        with open(snapshot_path, "w") as f:
            f.write("{}")
    with open(snapshot_path) as f:
        snapshot = json.load(f)

    snapshot.setdefault(key, {})
    snapshot[key][prompt] = hf_output_str

    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", [DTYPE])
@pytest.mark.parametrize("max_tokens", [MAX_TOKENS])
def test_models(
    hf_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    print("Running", model, dtype, max_tokens)
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    for i in range(len(example_prompts)):
        _, hf_output_str = hf_outputs[i]

        update_snapshot(model, dtype, max_tokens, example_prompts[i],
                        hf_output_str)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
