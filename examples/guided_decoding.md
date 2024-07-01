**Guided Decoding for Large Language Models (LLMs)**
=====================================================

Large Language Models (LLMs) have revolutionized the field of natural language processing, enabling applications such as text generation, language translation, and chatbots. However, one of the challenges of working with LLMs is controlling the output to meet specific requirements or constraints. This is where guided decoding comes in.

Guided decoding is a technique that allows you to influence the output of an LLM by providing additional context or constraints. By specifying a guiding mechanism, you can steer the model's output towards a desired format, structure, or content. This can be particularly useful in applications where the output needs to conform to specific rules, formats, or regulations.

In this example page, we will explore five guided decoding options for LLMs:

* **[Guided JSON](#guided-json)**: Constrain the output to follow a specific JSON schema.
* **[Guided Regex](#guided-regex)**: Restrict the output to match a regular expression pattern.
* **[Guided Choice](#guided-choice)**: Limit the output to a predefined set of choices.
* **[Guided Grammar](#guided-grammar)**: Conform the output to a context-free grammar.
* **[Guided Input](#guided-prompt)**: Lead the LLM by giving it the first words of its response.

These guided decoding options can help you unlock the full potential of LLMs and create more accurate, informative, and useful outputs. Let's dive into each of these options and explore how they can be used in practice.

## Guided Decoding Theory ##

Lets say you have a classification problem. You want to classify sentences as positive or negative. You ask the question: "What connotation does this sentence have?" The LLM generates token probabilities of:

* Positive: 0.5
* Negative: 0.1
* Sure: 0.2
* Let: 0.1
* All others: 0.1

This poses a problem because if it does select one of the other possibilites, downstream processes will break. It is possible to prompt engineer this with varying success, but there is a better option.

**Guided Decoding with Guided Choice:**

By applying guided decoding with guided choice, we can adjust the sampling process to only consider the relevant options: Positive or Negative.

In this case, the LLM's output would be sampled from a restricted probability distribution, where:

* Positive: 0.83 (rescaled from 0.5)
* Negative: 0.17 (rescaled from 0.1)
* Sure: 0.0 (excluded from sampling)
* Let: 0.0 (excluded from sampling)
* All others: 0.0 (excluded from sampling)

By guiding the output towards the desired options, we can increase the accuracy and relevance of the response ensure reliability on downstream tasks.

In this demo, we primarily use the OpenAI endpoint with the Outlines library. However, many of these concepts are applicable with the offline LLM class and/or post requests from the command line. For more information, see
* [VLLM OpenAI Documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters)
* [OpenAI Documentation](https://platform.openai.com/docs/api-reference/introduction)
* [Outlines Github](https://github.com/outlines-dev/outlines), [Outlines Documentation](https://outlines-dev.github.io/outlines/reference/)
* [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) - Alternative to outlines


```python
from openai import OpenAI
model = 'http://10.72.5.190:5006/v1'

client = OpenAI(
    api_key="EMPTY",
    base_url=model,
)
model_name = client.models.list().data[0].id # Grab the model name from the API
```

## Guided JSON <a id='guided-json'></a>

Outlines can make any open source model return a JSON object that follows a structure that is specified by the user. This is useful whenever we want the output of the model to be processed by code downstream: code does not understand natural language but rather the structured language it has been programmed to understand. 

Source: [Outlines Reference](https://outlines-dev.github.io/outlines/reference/json/)

JSON Schemas: [JSON Schema](https://json-schema.org/learn/getting-started-step-by-step)

To start, we need to specify the JSON schema. This may be done with the `pydantic` package and the `BaseModel`. Here, we are able to simply define the variables in the JSON as well as their types. `pydantic` converts this class into the schema that the LLM may use.


```python
import json
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user_schema = User.model_json_schema()
print(json.dumps(user_schema, indent=4))
```

    {
        "properties": {
            "name": {
                "title": "Name",
                "type": "string"
            },
            "age": {
                "title": "Age",
                "type": "integer"
            }
        },
        "required": [
            "name",
            "age"
        ],
        "title": "User",
        "type": "object"
    }
    

Using this schema, we may construct the model prompt. Here, we give the model our instruction, the schema, and the sentence. Note that while outlines does force the output to conform to the JSON schema, it is still useful to tell the model what that schema actually is. Remember, at their core, LLMs are next-token predictors, they do well when the tokens you want are easy to understand with the information given.

Using this prompt, we construct the messages object and call the LLM using the client object from earlier. Note that we include the `user_schema` as part of the `guided_json` in the `extra_body`. You may do a similar operation with curl or offline LLM.


```python
prompt = f"""
Convert the following sentence into a JSON object that conforms to the following schema:

{json.dumps(user_schema, indent=4)}

 Sentence: "John is 30 years old."

Please provide a JSON object with the correct attributes and values.
"""
messages = [{"role": "user", "content": prompt}]

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages)  # Chat history)

print('Without Guided Decoding')
print(output.choices[0].message.content)
print()

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages,  # Chat history
        extra_body={
            'guided_json':user_schema
        })

print('With Guided Decoding')
print(output.choices[0].message.content)


```

    Without Guided Decoding
    {
        "User": {
            "name": "John",
            "age": 30
        }
    }
    
    With Guided Decoding
    {"name": "John", "age": 30}
    

Note that without the guided decoding, the LLM included an extra "User" portion in the response. This would cause problems with downstream tasks. Meanwhile, the guided decoding outputted a json that is easily parsed without error.

## Guided Regex <a id='guided-regex'></a>
Regular expressions (regex) provide a powerful way to match and validate patterns in text. Guided Regex allows you to specify a regex pattern as a guiding mechanism, constraining the generated text to match the defined pattern. By providing a regex pattern, you can control the output of the language model and generate text that conforms to specific formats, such as dates, times, phone numbers, or other structured data.

Source: [Outlines Reference](https://outlines-dev.github.io/outlines/reference/regex/)

**Note** The backend cannot currently support all types of regex characters. Some characters such as `^`, `\A` and `$` are not supported and will return an error. In general, it seems like only single line outputs work.


```python
# regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)" # More complex example which includes full possible range.
regex = r'\d\.\d\.\d\.\d'
prompt = 'What is the IP address of the Google DNS servers?'
messages = [{"role": "user", "content": prompt}]

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages)  # Chat history)

print('Without Guided Decoding')
print(output.choices[0].message.content)
print()

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages,  # Chat history
        extra_body={
            'guided_regex':regex
        })

print('With Guided Decoding')
print(output.choices[0].message.content)
```

    Without Guided Decoding
    The IP addresses of Google's DNS servers are:
    
    ```
    8.8.8.8
    8.8.4.4
    ```
    
    These IP addresses can be used as a primary or secondary DNS server in your device's network settings.
    
    With Guided Decoding
    8.8.8.8
    

## Guided Choice <a id='guided-choice'></a>

In some cases, you may want to limit the output of the language model to a specific set of options or choices. Guided Choice allows you to specify a list of acceptable options, and the model will generate text that selects one of the provided choices. This guiding mechanism is particularly useful when you need to generate text that conforms to a specific set of categories, labels, or classifications.

Source: [Outlines Reference](https://outlines-dev.github.io/outlines/reference/choices/)

Let's consider the example from the intro. We have a bunch of sentences that we want to classify as positive or negative. Without guided decoding, the model gives a long response that would be nearly impossible to parse. With guided decoding, the model gives a single word answer that is easy to use in downstream tasks.


```python

prompt = 'What is the connotation of the following sentence, positive or negative?\n "I like puppies and rainbows"'
messages = [{"role": "user", "content": prompt}]

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages)  # Chat history)

print('Without Guided Decoding')
print(output.choices[0].message.content)
print()

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages,  # Chat history
        extra_body={
            'guided_choice':['Positive', 'Negative']
        })

print('With Guided Decoding')
print(output.choices[0].message.content)
```

    Without Guided Decoding
    The connotation of the sentence "I like puppies and rainbows" is generally considered positive. The words "puppies" and "rainbows" are often associated with feelings of happiness, innocence, and optimism. However, the interpretation may vary depending on the context in which the sentence is used. In some cases, it might be used sarcastically or in a nonsensical way, which could imply a more neutral or even slightly negative tone. But in general, the statement conveys a positive sentiment.
    
    With Guided Decoding
    Positive
    

## Guided Grammar <a id='guided-grammar'></a>

Guided Grammar is a guided decoding technique that allows you to constrain the output of a Large Language Model (LLM) to conform to a specific context-free grammar. This means you can define a set of rules that dictate the structure and syntax of the output, ensuring that it adheres to a particular format or style.

By specifying a guided grammar, you can influence the LLM's output to generate text that meets specific grammatical requirements, such as sentence structure, part-of-speech tags, or even domain-specific terminology. This can be particularly useful in applications where the output needs to conform to specific formatting or stylistic guidelines, such as generating code, writing reports, or creating content that meets specific regulatory requirements.

For example, you might use guided grammar to ensure that an LLM generates sentences with a specific subject-verb-object word order, or to restrict the output to a specific domain-specific vocabulary. By guiding the output towards a specific grammatical structure, you can increase the accuracy, relevance, and overall quality of the response.

Source: [Outlines Reference](https://outlines-dev.github.io/outlines/reference/cfg/)


```python
arithmetic_grammar = """
    ?start: expression

    ?expression: term (("+" | "-") term)*

    ?term: factor (("*" | "/") factor)*

    ?factor: NUMBER
           | "-" factor
           | "(" expression ")"

    %import common.NUMBER
"""

prompt = "Alice had 4 apples and Bob ate 2. "+ "Write an expression for Alice's apples:"
messages = [{"role": "user", "content": prompt}]

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages)  # Chat history)

print('Without Guided Decoding')
print(output.choices[0].message.content)
print()

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages,  # Chat history
        extra_body={
            'guided_grammar':arithmetic_grammar
        })

print('With Guided Decoding')
print(output.choices[0].message.content)
```

    Without Guided Decoding
    If Alice originally had 4 apples and Bob ate 2, then Alice's remaining apples can be represented by the expression:
    
    \[ \text{Alice's apples} = 4 - 2 \]
    
    This expression shows that Alice started with 4 apples and subtracted the 2 apples that Bob ate.
    
    With Guided Decoding
    4-2
    

## Guided Input <a id='guided-prompt'></a>

While traditional guided decoding methods focus on constraining the language model's output, primed input takes a different approach. By providing a few words or a short phrase to get the model started, you can influence the direction and tone of the generated text. This technique is not a traditional guided decoding method, but rather a way to provide a gentle nudge to the language model, helping it to generate text that is more relevant and coherent. By priming the input, you can encourage the model to explore specific topics, adopt certain styles, or convey particular emotions.

Here, we use the normal completions endpoint. As we are using Qwen-2 in this example, we use the ChatML format. For other models, refer to the model documentation. In this example, we are asking how to propose to our girlfriend. Without this formatting, the model starts with a short talk whereas with this formatting, the model knows to go directly into the suggestions.


```python
prompt = 'Give me 3 ways to propose to my girlfriend'
messages = [{"role": "user", "content": prompt}]

output = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=messages)  # Chat history)

print('Without Guided Decoding')
print(output.choices[0].message.content)
print()

# messages.append({"role": "assistant", "content": '1.)'})
response_start = '1.)'
prompt = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me 3 ways to propose to my girlfriend<|im_end|>\n<|im_start|>assistant\n{response_start}'
output = client.completions.create(
        model=model_name,  # Model name to use
        prompt=prompt,  # Chat history
        max_tokens=300
)

print('With Guided Decoding')
print(response_start + output.choices[0].text)
```

    Without Guided Decoding
    Proposing to your girlfriend is a special moment that you should tailor to your relationship, her personality, and your own personal style. Here are three unique and heartfelt ways you could propose:
    
    1. **Personalized Video Compilation**: Create a video that captures the highlights of your relationship, from the first date to your favorite moments together. Include clips, photos, and video messages from friends and family who love you both. At the end of the video, you could say something like, "I've put together this video to remind us of all the wonderful moments we've shared. But this moment is special. Do you want to spend the rest of our lives creating more memories together?" Then, drop to one knee and propose. This way, she'll see the love and effort you've put into the proposal and feel deeply connected to the moment.
    
    2. **Adventure Proposal**: If you both love nature, adventure, or a thrilling experience, you could propose during a special activity. For example, you could plan a hot air balloon ride where you have the whole basket to yourselves. As you ascend, you could suggest she step out on the basket edge for a photo (with your camera ready) and, once she is safely back in, propose. Alternatively, you could plan a hike to a scenic viewpoint where you have set up a picnic with a special message or even a small ring box. The surprise and the beautiful setting can make the moment even more memorable.
    
    3. **Surprise Dinner and Proposal**: Plan a special dinner at your favorite restaurant or, if you prefer, cook a gourmet meal at home. Make it a surprise and let her know that you have a reservation at a quiet table or have set the dining table with candles, flowers, and your favorite music playing in the background. During the dinner, you could tell her about a special memory or milestone in your relationship, then lead her to believe you're sharing a toast. As you raise your glass, you could say something like, "To us, and to all the beautiful moments we have shared, and to the future we will create together." On a tray or a nearby table, place the ring box, and then pop the question. This approach offers a blend of romance and intimacy.
    
    Remember, the most important aspect of proposing is showing your genuine love and commitment to your partner. Tailor the proposal to reflect your relationship and ensure it is a moment that you both will cherish forever.
    
    With Guided Decoding
    1.) Speak to her in person: This is perhaps the most romantic way to ask your girl friend to marry you. Getting in a romantic setting and expressing your love and commitment to her publicly can create a memorable moment you and she can cherish.
    
    2.) Write her a love letter: There is no better way to express your feelings for her and ask her to marry you than with a hand-written love letter. This type of proposal shows her that you take the commitment seriously and cherish her enough to put your feelings into words.
    
    3.) Plan a surprise proposal: Consider a romantic location that you both love, such as a favorite place in your hometown, a special attraction, or a place with sentimental value. Plan a proposal that matches your love story. Be sure to keep it a secret from her as not to spoil the surprise.
    
