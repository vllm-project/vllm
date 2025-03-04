# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from openai import BadRequestError, OpenAI
from pydantic import BaseModel

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)

# Guided decoding by Choice (list of possible options)
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": "Classify this sentiment: vLLM is wonderful!"
    }],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)

# Guided decoding by Regex
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={
        "guided_regex": "\w+@\w+\.com\n",
        "stop": ["\n"]
    },
)
print(completion.choices[0].message.content)


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

prompt = ("Generate a JSON with the brand, model and car_type of"
          "the most iconic car from the 90's")
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_json": json_schema},
)
print(completion.choices[0].message.content)

# Guided decoding by Grammar
simplified_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""

prompt = ("Generate an SQL query to show the 'username' and 'email'"
          "from the 'users' table.")
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    extra_body={"guided_grammar": simplified_sql_grammar},
)
print(completion.choices[0].message.content)

# Extra backend options
prompt = ("Generate an email address for Alan Turing, who works in Enigma."
          "End in .com and new line. Example result:"
          "alan.turing@enigma.com\n")

try:
    # The no-fallback option forces vLLM to use xgrammar, so when it fails
    # you get a 400 with the reason why
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        extra_body={
            "guided_regex": "\w+@\w+\.com\n",
            "stop": ["\n"],
            "guided_decoding_backend": "xgrammar:no-fallback"
        },
    )
except BadRequestError as e:
    print("This error is expected:", e)
