# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from pydantic import BaseModel

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", max_model_len=100)

# Guided decoding by Choice (list of possible options)
guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
outputs = llm.generate(
    prompts="Classify this sentiment: vLLM is wonderful!",
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)

# Guided decoding by Regex
guided_decoding_params = GuidedDecodingParams(regex="\w+@\w+\.com\n")
sampling_params = SamplingParams(guided_decoding=guided_decoding_params,
                                 stop=["\n"])
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")
outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)


# Guided decoding by JSON using Pydantic schema
class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


json_schema = CarDescription.model_json_schema()

guided_decoding_params = GuidedDecodingParams(json=json_schema)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
prompt = ("Generate a JSON with the brand, model and car_type of"
          "the most iconic car from the 90's")
outputs = llm.generate(
    prompts=prompt,
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)

# Guided decoding by Grammar
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""
guided_decoding_params = GuidedDecodingParams(grammar=simplified_sql_grammar)
sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
prompt = ("Generate an SQL query to show the 'username' and 'email'"
          "from the 'users' table.")
outputs = llm.generate(
    prompts=prompt,
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
