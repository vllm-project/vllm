import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "internlm/internlm2-1_8b"

PROMPT_TEMPLATE = "[user] question: {sql_prompt}\n\n context: {context}\n\n [/user] [assistant] "  # noqa: E501


def do_sample(llm, lora_path: str, lora_id: int) -> str:
    prompts = [
        PROMPT_TEMPLATE.format(
            sql_prompt=
            "Which catalog publisher has published the most catalogs?",
            context="CREATE TABLE catalogs (catalog_publisher VARCHAR);"),
        PROMPT_TEMPLATE.format(
            sql_prompt=
            "Remove the 'vehicle_safety_testing' table and its records.",  # noqa: E501
            context=
            "CREATE TABLE vehicle_safety_testing (id INT PRIMARY KEY, vehicle_model VARCHAR(255), test_score FLOAT);"  # noqa: E501
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
                                          stop="[/assistant]")
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


def test_internlm2_lora(internlm2_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(MODEL_PATH,
                   trust_remote_code=True,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=2,
                   enforce_eager=True)

    expected_lora_output = [
        "SELECT catalog_publisher, COUNT(*) as num_catalogs FROM catalogs GROUP BY catalog_publisher ORDER BY num_catalogs DESC LIMIT 1;",  # noqa: E501
        "DROP TABLE vehicle_safety_testing;",  # noqa: E501
        "SELECT COUNT(*) FROM marine_species WHERE location = 'Southern Ocean';",  # noqa: E501
    ]

    output1 = do_sample(llm, internlm2_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
    output2 = do_sample(llm, internlm2_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i].startswith(expected_lora_output[i])
