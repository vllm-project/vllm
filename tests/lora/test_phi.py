import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "microsoft/phi-2"

PROMPT_TEMPLATE = "### Instruct: {sql_prompt}\n\n### Context: {context}\n\n### Output:"  # noqa: E501


def do_sample(llm, lora_path: str, lora_id: int) -> str:
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
            "What is the maximum temperature recorded in the 'arctic_weather' table for each month in the year 2020, broken down by species ('species' column in the 'arctic_weather' table)?",  # noqa: E501
            context=
            "CREATE TABLE arctic_weather (id INT, date DATE, temperature FLOAT, species VARCHAR(50));"  # noqa: E501
        ),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32, stop="### End")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None,
    )
    # Print the outputs.
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def test_phi2_lora(phi2_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4)

    expected_lora_output = [
        " SELECT catalog_publisher, COUNT(*) as count FROM catalogs GROUP BY catalog_publisher ORDER BY count DESC;"  # noqa: E501
        " SELECT t.id FROM trip t JOIN station s ON t.start_station_id = s.id WHERE s.dock_count = (SELECT MAX(dock_count) FROM station);"  # noqa: E501
        " SELECT species, MAX(temperature) AS max_temperature FROM arctic_weather WHERE YEAR(date) = 2020 GROUP BY species;",  # noqa: E501
    ]

    output1 = do_sample(llm, phi2_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
    output2 = do_sample(llm, phi2_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i].startswith(expected_lora_output[i])
