# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "microsoft/phi-2"

PROMPT_TEMPLATE = "### Instruct: {sql_prompt}\n\n### Context: {context}\n\n### Output:"  # noqa: E501


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    prompts = [
        PROMPT_TEMPLATE.format(
            sql_prompt=
            "Which catalog publisher has published the most catalogs?",
            context="CREATE TABLE catalogs (catalog_publisher VARCHAR);"),
        PROMPT_TEMPLATE.format(
            sql_prompt=
            "Which trip started from the station with the largest dock count? Give me the trip id.",  # noqa: E501
            context=
            "CREATE TABLE trip (id VARCHAR, start_station_id VARCHAR); CREATE TABLE station (id VARCHAR, dock_count VARCHAR);"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            sql_prompt=
            "How many marine species are found in the Southern Ocean?",  # noqa: E501
            context=
            "CREATE TABLE marine_species (name VARCHAR(50), common_name VARCHAR(50), location VARCHAR(50));"  # noqa: E501
        ),
    ]
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=64,
                                          stop="### End")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None,
    )
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def test_phi2_lora(phi2_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=2,
                   enforce_eager=True,
                   enable_chunked_prefill=True)

    expected_lora_output = [
        "SELECT catalog_publisher, COUNT(*) as num_catalogs FROM catalogs GROUP BY catalog_publisher ORDER BY num_catalogs DESC LIMIT 1;",  # noqa: E501
        "SELECT trip.id FROM trip JOIN station ON trip.start_station_id = station.id WHERE station.dock_count = (SELECT MAX(dock_count) FROM station);",  # noqa: E501
        "SELECT COUNT(*) FROM marine_species WHERE location = 'Southern Ocean';",  # noqa: E501
    ]

    output1 = do_sample(llm, phi2_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
    output2 = do_sample(llm, phi2_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i].startswith(expected_lora_output[i])
