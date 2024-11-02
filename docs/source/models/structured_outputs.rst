.. _structured_outputs:

Structured Outputs
==================

vLLM supports the generation of structured outputs using `outlines <https://github.com/dottxt-ai/outlines>`_ or `lm-format-enforcer <https://github.com/noamgat/lm-format-enforcer>`_ as backends for the guided decoding.
This document shows you some examples of the different options that are available to generate structured outputs. 


Online Inference (OpenAI API)
-----------------------------

You can generate structured outputs using the OpenAI’s `Completions <https://platform.openai.com/docs/api-reference/completions>`_ and `Chat <https://platform.openai.com/docs/api-reference/chat>`_  API.

The following parameters are supported, which must be added as extra parameters:

- ``guided_choice``: the output will be exactly one of the choices.
- ``guided_regex``: the output will follow the regex pattern.
- ``guided_json``: the output will follow the JSON schema.
- ``guided_grammar``: the output will follow the context free grammar.
- ``guided_whitespace_pattern``: used to override the default whitespace pattern for guided json decoding.
- ``guided_decoding_backend``: used to select the guided decoding backend to use.

Now let´s see an example for each of the cases, starting with the ``guided_choice``, as it´s the easiest one: 

.. code-block:: python

    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="-",
    )

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B",
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
        ],
        extra_body={"guided_choice": ["positive", "negative"]},
    )
    print(completion.choices[0].message.content)


The next example shows how to use the ``guided_regex``. The idea is to generate an email address, given a simple regex template: 

.. code-block:: python

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B",
        messages=[
            {
                "role": "user",
                "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
            }
        ],
        extra_body={"guided_regex": "\w+@\w+\.com\n", "stop": ["\n"]},
    )
    print(completion.choices[0].message.content)

One of the most relevant features in structured text generation is the option to generate a valid JSON with pre-defined fields and formats. 
For this we can use the ``guided_json`` parameter in two different ways:

- Using directly a `JSON Schema <https://json-schema.org/>`_ 
- Defining a `Pydantic model <https://docs.pydantic.dev/latest/>`_ and then extracting the JSON Schema from it (which is normally an easier option).

The next example shows how to use the ``guided_json`` parameter with a Pydantic model:

.. code-block:: python

    from pydantic import BaseModel
    from enum import Enum

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

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B",
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
            }
        ],
        extra_body={"guided_json": json_schema},
    )
    print(completion.choices[0].message.content)

.. tip::
    While not strictly necessary, normally it´s better to indicate in the prompt that a JSON needs to be generated and which fields and how should the LLM fill them.
    This can improve the results notably in most cases.


Finally we have the ``guided_grammar``, which probably is the most difficult one to use but it´s really powerful, as it allows us to define complete languages like SQL queries.
It works by using a context free EBNF grammar, which for example we can use to define a specific format of simplified SQL queries, like in the example below:

.. code-block:: python

    simplified_sql_grammar = """
        ?start: select_statement

        ?select_statement: "SELECT " column_list " FROM " table_name

        ?column_list: column_name ("," column_name)*

        ?table_name: identifier

        ?column_name: identifier

        ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
    """

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B",
        messages=[
            {
                "role": "user",
                "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
            }
        ],
        extra_body={"guided_grammar": simplified_sql_grammar},
    )
    print(completion.choices[0].message.content)

The complete code of the examples can be found on `examples/openai_chat_completion_structured_outputs.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_chat_completion_structured_outputs.py>`_.


Offline Inference
-----------------

Offline inference allows for the same types of guided decoding.
To use it, we´ll need to configure the guided decoding using the class ``GuidedDecodingParams`` inside ``SamplingParams``. 
The main available options inside ``GuidedDecodingParams`` are: 

- ``json`` 
- ``regex`` 
- ``choice``
- ``grammar``
- ``backend``
- ``whitespace_pattern``

These parameters can be used in the same way as the parameters from the Online Inference examples above. 
One example for the usage of the ``choices`` parameter is shown below: 

.. code-block:: python

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    guided_decoding_params = GuidedDecodingParams(choice=["Positive", "Negative"])
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: vLLM is wonderful!",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)

A complete example with all options can be found in `examples/offline_inference_structured_outputs.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_structured_outputs.py>`_.