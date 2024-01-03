"""
Inspired by the OpenAI example found here:
    https://platform.openai.com/docs/guides/function-calling/parallel-function-calling
"""

import datetime
from openai import OpenAI
import json

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
models = client.models.list()
model = models.data[0].id
stream = True


def get_current_date_utc():
    print("Calling get_current_date_utc client side.")
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "The current UTC datetime is (day: %A, date (day/month/year): %d/%m/%Y, time: %H:%M)."
    )


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    print("Calling get_current_weather client side.")
    if "tokyo" in location.lower():
        return json.dumps({
            "location": "Tokyo",
            "temperature": "10",
            "unit": unit
        })
    elif "san francisco" in location.lower():
        return json.dumps({
            "location": "San Francisco",
            "temperature": "72",
            "unit": unit
        })
    elif "paris" in location.lower():
        return json.dumps({
            "location": "Paris",
            "temperature": "22",
            "unit": unit
        })
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def run_conversation():
    # Step 1: send the conversation and available functions to the model
    # messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    messages = [{
        "role":
        "user",
        "content":
        "What's the weather like in San Francisco, Tokyo, and Paris ? We also need to know the current date."
    }]
    tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type":
                        "string",
                        "description":
                        "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        },
    }, {
        "type": "function",
        "function": {
            "name": "get_current_date_utc",
            "description": "Get the current UTC time",
        },
    }]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = ""
    tool_calls = None
    if stream:
        text_message = ""
        for chunk in response:
            if chunk.choices[0].finish_reason is not None:
                if chunk.choices[0].finish_reason == "tool_calls":
                    tool_calls = chunk.choices[0].delta.tool_calls
                break
            if chunk.choices[0].delta.content is not None:
                text_message += chunk.choices[0].delta.content
        response_message = {"role": "assistant", "content": text_message}
    else:
        if not len(response.choices):
            return None
        response_message = response.choices[0].message
        # print(str(response_message))
        tool_calls = response_message.tool_calls

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
            "get_current_date_utc": get_current_date_utc,
        }
        messages.append(
            response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            if function_name == "get_current_weather":
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
            else:
                function_response = function_to_call()

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )  # get a new response from the model where it can see the function response

        for it_msg, msg in enumerate(messages):
            print("Message %i:\n    %s\n" % (it_msg, str(msg)))

        return second_response


result = run_conversation()
print("Final response:\n%s" % result)
