from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")

llm = LLM(model="meta-llama/Llama-2-7b-hf",
          enable_lora=True,
          max_num_seqs=2,
          dtype='bfloat16')

sampling_params = SamplingParams(temperature=0,
                                 max_tokens=1024,
                                 stop=["[/assistant]"])

prompts = [
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_95 (one_mora VARCHAR, gloss VARCHAR, accented_mora VARCHAR)\n\n question: What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant]",  # noqa: E501
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE candidate (people_id VARCHAR, unsure_rate INTEGER); CREATE TABLE people (sex VARCHAR, people_id VARCHAR)\n\n question: which gender got the highest average uncertain ratio. [/user] [assistant]",  # noqa: E501
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_60 (pick INTEGER, former_wnba_team VARCHAR)\n\n question: What pick was a player that previously played for the Minnesota Lynx? [/user] [assistant]",  # noqa: E501
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_28138035_4 (womens_doubles VARCHAR, mens_singles VARCHAR)\n\n question: Name the women's doubles for werner schlager [/user] [assistant]"  # noqa: E501
]

expected_output = [
    "  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' ",  # noqa: E501
    "  SELECT nationality FROM table_name_11 WHERE elector = 'Anchero Pantaleone' ",  # noqa: E501
    "  SELECT one_mora FROM table_name_95 WHERE gloss = 'low tone mora with a gloss of /˩okiru/' [òkìɽɯ́] AND accented_mora = 'low tone mora with a gloss of /˩okiru/' [òkìɽɯ́] ",  # noqa: E501
    "  SELECT sex FROM people WHERE people_id IN (SELECT people_id FROM candidate GROUP BY sex ORDER BY COUNT(people_id) DESC LIMIT 1) ",  # noqa: E501
    "  SELECT pick FROM table_name_60 WHERE former_wnba_team = 'Minnesota Lynx' ",  # noqa: E501
    "  SELECT womens_doubles FROM table_28138035_4 WHERE mens_singles = 'Werner Schlager' "  # noqa: E501
]

outputs = llm.generate(prompts,
                       sampling_params,
                       lora_request=LoRARequest("sql_adapter", 1,
                                                sql_lora_path))

for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    match = expected_output[i] == generated_text
    if not match:
        print(
            f"Comparison failed for request_id::{i}\n\t[PROMPT]{prompt!r}\n\t[GENERATED]{generated_text!r}\n\t[EXPECTED]{expected_output[i]!r}"  # noqa: E501
        )
