from typing import List

import os
import pytest
import ray

import vllm
from vllm.lora.request import LoRARequest

#from .conftest import cleanup

MODEL_PATH = "/workspace/meta-llama3-8b-instruct"


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_95 (one_mora VARCHAR, gloss VARCHAR, accented_mora VARCHAR)\n\n question: What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE candidate (people_id VARCHAR, unsure_rate INTEGER); CREATE TABLE people (sex VARCHAR, people_id VARCHAR)\n\n question: which gender got the highest average uncertain ratio. [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_60 (pick INTEGER, former_wnba_team VARCHAR)\n\n question: What pick was a player that previously played for the Minnesota Lynx? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_28138035_4 (womens_doubles VARCHAR, mens_singles VARCHAR)\n\n question: Name the women's doubles for werner schlager [/user] [assistant]"  # noqa: E501
    ]
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=128)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts: List[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"\nPrompt: {prompt!r}, \nGenerated text: {generated_text!r}\n")
    return generated_texts

def compare_strings(str1, str2):
    if str1 == str2:
        return True
    else:
        print("Strings are different:")
        min_length = min(len(str1), len(str2))
        if len(str1) != len(str2):
            print(f"Length difference: {len(str1)} vs {len(str2)}")
        w=2
        for i in range(min_length):
            if str1[i] != str2[i]:
                print(f"Difference at index {i}: '{str1[i-w:i+w]}' vs '{str2[i-w:i+w]}'")
                return False
        return False


if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    
    tp_size=1
    num_gpus_available=1
    
    llama3_lora_lm_head_files='/workspace/llama3_lora'

    if num_gpus_available < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    llm = vllm.LLM(MODEL_PATH,
                   #enable_lora=True,
                   #max_num_seqs=16,
                   #max_loras=4,
                   gpu_memory_utilization=0.5,
                   tensor_parallel_size=tp_size)

    expected_no_lora_output = [
    " \n\nHere is the SQL query to answer the question:\n\n```sql\nSELECT icao\nFROM table_name_74\nWHERE airport = 'Lilongwe International Airport';\n```\n\nThis query will return the ICAO code for Lilongwe International Airport. If there are multiple rows with the same airport name, this query will return all of them. If you want to return only one row, you can add a UNIQUE constraint to the airport column in the table schema. [/assistant] \n\nNote: The above query assumes that the airport name is exactly 'Lilongwe International Airport'. If the airport name can be in different",
    " \n\nHere is the SQL query to answer the question:\n\n```sql\nSELECT nationality\nFROM table_name_11\nWHERE elector = 'Anchero Pantaleone';\n```\n\nThis query will return the nationality of Anchero Pantaleone when he was the elector. If there are multiple rows with the same elector, this query will return all of them. If you want to return only one row, you can add a LIMIT clause:\n\n```sql\nSELECT nationality\nFROM table_name_11\nWHERE elector = 'Anchero Pantaleone'\nLIMIT 1;\n``` [/assistant] \n\nNote: The above",
    " To answer this question, we need to find the row in the table where the `accented_mora` column matches the low tone mora with the gloss `/˩okiru/ [òkìɽɯ́]`. \n\nHere is the SQL query to do that:\n\n```sql\nSELECT one_mora\nFROM table_name_95\nWHERE accented_mora = '/˩okiru/ [òkìɽɯ́]';\n```\n\nThis query will return the value in the `one_mora` column for the row where the `accented_mora` column matches the specified",
    " To answer this question, we need to join the `candidate` table with the `people` table on the `people_id` column, and then calculate the average uncertain ratio for each gender. Here is the SQL query to do this:\n\n```sql\nSELECT p.sex, AVG(c.unsure_rate) AS avg_unsure_rate\nFROM candidate c\nJOIN people p ON c.people_id = p.people_id\nGROUP BY p.sex\nORDER BY avg_unsure_rate DESC;\n```\n\nThis query will return the gender with the highest average uncertain ratio. If there are multiple genders with the same highest average uncertain ratio, this query will return",
    " To answer this question, we can use a SQL query that joins the table with itself to find the players who previously played for the Minnesota Lynx and then filter the results to only include the pick information. Here is the SQL query:\n\n```sql\nSELECT t1.pick\nFROM table_name_60 t1\nJOIN table_name_60 t2 ON t1.former_wnba_team = t2.former_wnba_team\nWHERE t2.former_wnba_team = 'Minnesota Lynx';\n```\n\nThis query works by joining the table with itself on the `former_wnba_team` column. The `",
    " \n\nHere is the SQL query to answer the question:\n\n```sql\nSELECT womens_doubles \nFROM table_28138035_4 \nWHERE mens_singles = 'Werner Schlager';\n```\n\nThis query will return the women's doubles for Werner Schlager. If there are multiple rows with the same mens_singles value, this query will return all of them. If you want to return only one row, you can add a LIMIT clause:\n\n```sql\nSELECT womens_doubles \nFROM table_28138035_4 \nWHERE mens_singles = 'Werner Schlager'\nLIMIT 1;\n``` [/assistant] \n\n"
    ]

    expected_lora_output = [
        "  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' ",  # noqa: E501
        "  SELECT nationality FROM table_name_11 WHERE elector = 'anchero pantaleone' ",  # noqa: E501
        "  SELECT one_mora FROM table_name_95 WHERE gloss = 'low tone mora with a gloss of /˩okiru/' [òkìɽɯ́] AND accented_mora = 'low tone mora with a gloss of /˩okiru/' [òkìɽɯ́] ",  # noqa: E501
        "  SELECT sex FROM people WHERE people_id IN (SELECT people_id FROM candidate GROUP BY sex ORDER BY COUNT(people_id) DESC LIMIT 1) ",  # noqa: E501
        "  SELECT pick FROM table_name_60 WHERE former_wnba_team = 'Minnesota Lynx' ",  # noqa: E501
        "  SELECT womens_doubles FROM table_28138035_4 WHERE mens_singles = 'Werner Schlager' "  # noqa: E501
    ]

    print("no lora")
    no_lora_output=do_sample(llm, llama3_lora_lm_head_files, lora_id=0)
    for i, (a,e) in enumerate(zip(no_lora_output, expected_no_lora_output)):
        if not compare_strings(a,e):
            print(f"{i}-th string is different")

    print("lora 1")
    assert do_sample(llm, llama3_lora_lm_head_files, lora_id=1) == expected_lora_output
