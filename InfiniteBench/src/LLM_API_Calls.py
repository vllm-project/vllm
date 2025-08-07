# LLM_API_Calls.py
#########################################
# Commercial LLM API Calls Library
# This library is used to perform API calls to commercial API endpoints.
#
####
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. chat_with_openai(api_key, file_path, custom_prompt_arg)
# 3. chat_with_anthropic(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. chat_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. chat_with_groq(api_key, input_data, custom_prompt_arg, system_prompt=None):
# 6. chat_with_openrouter(api_key, input_data, custom_prompt_arg, system_prompt=None)
# 7. chat_with_huggingface(api_key, input_data, custom_prompt_arg, system_prompt=None)
# 8. chat_with_deepseek(api_key, input_data, custom_prompt_arg, system_prompt=None)
# 9. chat_with_vllm(input_data, custom_prompt_input, api_key=None, vllm_api_url="http://127.0.0.1:8000/v1/chat/completions", system_prompt=None)
#
#
####################
#
# Import necessary libraries
import json
import logging
import os
import time
import requests
#
# Import 3rd-Party Libraries
from requests import RequestException
#
# Import Local libraries
from eval_utils import load_and_log_configs
#
#######################################################################################################################
# Function Definitions
#

#FIXME: Update to include full arguments

def extract_text_from_segments(segments):
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")

    text = ""

    if isinstance(segments, list):
        for segment in segments:
            logging.debug(f"Current segment: {segment}")
            logging.debug(f"Type of segment: {type(segment)}")
            if 'Text' in segment:
                text += segment['Text'] + " "
            else:
                logging.warning(f"Skipping segment due to missing 'Text' key: {segment}")
    else:
        logging.warning(f"Unexpected type of 'segments': {type(segments)}")

    return text.strip()



def chat_with_openai(api_key, input_data, custom_prompt_arg, temp=None, system_message=None):
    loaded_config_data = load_and_log_configs()
    openai_api_key = api_key
    try:
        # API key validation
        if not openai_api_key:
            logging.info("OpenAI: API key not provided as parameter")
            logging.info("OpenAI: Attempting to use API key from config file")
            openai_api_key = loaded_config_data['api_keys']['openai']

        if not openai_api_key:
            logging.error("OpenAI: API key not found or is empty")
            return "OpenAI: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"OpenAI: Using API Key: {openai_api_key[:5]}...{openai_api_key[-5:]}")

        # Input data handling
        logging.debug(f"OpenAI: Raw input data type: {type(input_data)}")
        logging.debug(f"OpenAI: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("OpenAI: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"OpenAI: Error parsing JSON string: {str(e)}")
                    return f"OpenAI: Error parsing JSON input: {str(e)}"
            elif os.path.isfile(input_data):
                logging.debug("OpenAI: Loading JSON data from file for summarization")
                with open(input_data, 'r') as file:
                    data = json.load(file)
            else:
                logging.debug("OpenAI: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"OpenAI: Processed data type: {type(data)}")
        logging.debug(f"OpenAI: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("OpenAI: Summary already exists in the loaded data")
                return data['summary']
            elif 'segments' in data:
                text = extract_text_from_segments(data['segments'])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(f"OpenAI: Invalid input data format: {type(data)}")

        logging.debug(f"OpenAI: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"OpenAI: Custom prompt: {custom_prompt_arg}")

        openai_model = loaded_config_data['services']['openai'] or "gpt-4o"
        logging.debug(f"OpenAI: Using model: {openai_model}")

        headers = {
            'Authorization': f'Bearer {openai_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")
        logging.debug("openai: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        if temp is None:
            temp = 0.7
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."
        temp = float(temp)
        data = {
            "model": openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt}
            ],
            "max_tokens": 4096,
            "temperature": temp
        }

        logging.debug("OpenAI: Posting request")
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                chat_response = response_data['choices'][0]['message']['content'].strip()
                logging.debug("openai: Chat Sent successfully")
                return chat_response
            else:
                logging.warning("openai: Chat response not found in the response data")
                return "openai: Chat not available"
        else:
            logging.error(f"OpenAI: Chat request failed with status code {response.status_code}")
            logging.error(f"OpenAI: Error response: {response.text}")
            return f"OpenAI: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"OpenAI: Error decoding JSON: {str(e)}", exc_info=True)
        return f"OpenAI: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"OpenAI: Error making API request: {str(e)}", exc_info=True)
        return f"OpenAI: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"OpenAI: Unexpected error: {str(e)}", exc_info=True)
        return f"OpenAI: Unexpected error occurred: {str(e)}"


def chat_with_anthropic(api_key, input_data, model, custom_prompt_arg, max_retries=3, retry_delay=5, system_prompt=None):
    try:
        loaded_config_data = load_and_log_configs()
        global anthropic_api_key
        anthropic_api_key = api_key
        # API key validation
        if not api_key:
            logging.info("Anthropic: API key not provided as parameter")
            logging.info("Anthropic: Attempting to use API key from config file")
            anthropic_api_key = loaded_config_data['api_keys']['anthropic']

        if not api_key or api_key.strip() == "":
            logging.error("Anthropic: API key not found or is empty")
            return "Anthropic: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Anthropic: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        if system_prompt is not None:
            logging.debug("Anthropic: Using provided system prompt")
            pass
        else:
            system_prompt = "You are a helpful assistant"

        logging.debug(f"AnthropicAI: Loaded data: {input_data}")
        logging.debug(f"AnthropicAI: Type of data: {type(input_data)}")

        anthropic_model = loaded_config_data['services']['anthropic']

        headers = {
            'x-api-key': anthropic_api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        anthropic_user_prompt = custom_prompt_arg
        logging.debug(f"Anthropic: User Prompt is {anthropic_user_prompt}")
        user_message = {
            "role": "user",
            "content": f"{input_data} \n\n\n\n{anthropic_user_prompt}"
        }

        data = {
            "model": model,
            "max_tokens": 4096,  # max _possible_ tokens to return
            "messages": [user_message],
            "stop_sequences": ["\n\nHuman:"],
            "temperature": 0.1,
            "top_k": 0,
            "top_p": 1.0,
            "metadata": {
                "user_id": "example_user_id",
            },
            "stream": False,
            "system": f"{system_prompt}"
        }

        for attempt in range(max_retries):
            try:
                logging.debug("anthropic: Posting request to API")
                response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data)

                # Check if the status code indicates success
                if response.status_code == 200:
                    logging.debug("anthropic: Post submittal successful")
                    response_data = response.json()
                    try:
                        chat_response = response_data['content'][0]['text'].strip()
                        logging.debug("anthropic: Chat request successful")
                        print("Chat request processed successfully.")
                        return chat_response
                    except (IndexError, KeyError) as e:
                        logging.debug("anthropic: Unexpected data in response")
                        print("Unexpected response format from Anthropic API:", response.text)
                        return None
                elif response.status_code == 500:  # Handle internal server error specifically
                    logging.debug("anthropic: Internal server error")
                    print("Internal server error from API. Retrying may be necessary.")
                    time.sleep(retry_delay)
                else:
                    logging.debug(
                        f"anthropic: Failed to process chat request, status code {response.status_code}: {response.text}")
                    print(f"Failed to process chat request, status code {response.status_code}: {response.text}")
                    return None

            except RequestException as e:
                logging.error(f"anthropic: Network error during attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"anthropic: Network error: {str(e)}"
    except Exception as e:
        logging.error(f"anthropic: Error in processing: {str(e)}")
        return f"anthropic: Error occurred while processing summary with Anthropic: {str(e)}"


# Summarize with Cohere
def chat_with_cohere(api_key, input_data, model, custom_prompt_arg, system_prompt=None):
    global cohere_api_key
    cohere_api_key = api_key
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if not api_key:
            logging.info("cohere: API key not provided as parameter")
            logging.info("cohere: Attempting to use API key from config file")
            cohere_api_key = loaded_config_data['api_keys']['cohere']

        if not api_key or api_key.strip() == "":
            logging.error("cohere: API key not found or is empty")
            return "cohere: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"cohere: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        logging.debug(f"Cohere: Loaded data: {input_data}")
        logging.debug(f"Cohere: Type of data: {type(input_data)}")

        cohere_model = loaded_config_data['services']['cohere']

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {cohere_api_key}'
        }

        if system_prompt is not None:
            logging.debug("Anthropic: Using provided system prompt")
            pass
        else:
            system_prompt = "You are a helpful assistant"

        cohere_prompt = f"{input_data} \n\n\n\n{custom_prompt_arg}"
        logging.debug(f"cohere: User Prompt being sent is {cohere_prompt}")

        logging.debug(f"cohere: System Prompt being sent is {system_prompt}")

        data = {
            "chat_history": [
                {"role": "SYSTEM", "message": f"system_prompt"},
            ],
            "message": f"{cohere_prompt}",
            "model": model,
            "connectors": [{"id": "web-search"}]
        }

        logging.debug("cohere: Submitting request to API endpoint")
        print("cohere: Submitting request to API endpoint")
        response = requests.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)
        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'text' in response_data:
                chat_response = response_data['text'].strip()
                logging.debug("cohere: Chat request successful")
                print("Chat request processed successfully.")
                return chat_response
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"cohere: API request failed with status code {response.status_code}: {response.text}")
            print(f"Failed to process summary, status code {response.status_code}: {response.text}")
            return f"cohere: API request failed: {response.text}"

    except Exception as e:
        logging.error("cohere: Error in processing: %s", str(e))
        return f"cohere: Error occurred while processing summary with Cohere: {str(e)}"


# https://console.groq.com/docs/quickstart
def chat_with_groq(api_key, input_data, custom_prompt_arg, temp=None, system_message=None):
    logging.debug("Groq: Summarization process starting...")
    try:
        logging.debug("Groq: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            groq_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                groq_api_key = api_key
                logging.info("Groq: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                groq_api_key = loaded_config_data['api_keys'].get('groq')
                if groq_api_key:
                    logging.info("Groq: Using API key from config file")
                else:
                    logging.warning("Groq: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not groq_api_key or not groq_api_key.strip():
            logging.error("Anthropic: No valid API key available")
            # You might want to raise an exception here or handle this case as appropriate for your application
            # For example: raise ValueError("No valid Anthropic API key available")

        logging.debug(f"Groq: Using API Key: {groq_api_key[:5]}...{groq_api_key[-5:]}")

        # Transcript data handling & Validation
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Groq: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Groq: Using provided string data for summarization")
            data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"Groq: Loaded data: {data[:500]}...(snipped to first 500 chars)")
        logging.debug(f"Groq: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Groq: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Groq: Invalid input data format")

        # Set the model to be used
        groq_model = loaded_config_data['services']['groq']

        if temp is None:
            temp = 0.2
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {groq_api_key}',
            'Content-Type': 'application/json'
        }

        groq_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug("groq: Prompt being sent is {groq_prompt}")

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": groq_prompt,
                }
            ],
            "model": groq_model,
            "temperature": temp
        }

        logging.debug("groq: Submitting request to API endpoint")
        print("groq: Submitting request to API endpoint")
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data)

        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("groq: Chat request successful")
                print("Groq: Chat request successful.")
                return summary
            else:
                logging.error("Groq(chat): Expected data not found in API response.")
                return "Groq(chat): Expected data not found in API response."
        else:
            logging.error(f"groq: API request failed with status code {response.status_code}: {response.text}")
            return f"groq: API request failed: {response.text}"

    except Exception as e:
        logging.error("groq: Error in processing: %s", str(e))
        return f"groq: Error occurred while processing summary with groq: {str(e)}"


def chat_with_openrouter(api_key, input_data, custom_prompt_arg, temp=None, system_message=None):
    import requests
    import json
    global openrouter_model, openrouter_api_key
    try:
        logging.debug("OpenRouter: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            openrouter_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                openrouter_api_key = api_key
                logging.info("OpenRouter: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                openrouter_api_key = loaded_config_data['api_keys'].get('openrouter')
                if openrouter_api_key:
                    logging.info("OpenRouter: Using API key from config file")
                else:
                    logging.warning("OpenRouter: No API key found in config file")

        # Model Selection validation
        logging.debug("OpenRouter: Validating model selection")
        loaded_config_data = load_and_log_configs()
        openrouter_model = loaded_config_data['services']['openrouter']
        logging.debug(f"OpenRouter: Using model from config file: {openrouter_model}")

        # Final check to ensure we have a valid API key
        if not openrouter_api_key or not openrouter_api_key.strip():
            logging.error("OpenRouter: No valid API key available")
            raise ValueError("No valid Anthropic API key available")
    except Exception as e:
        logging.error("OpenRouter: Error in processing: %s", str(e))
        return f"OpenRouter: Error occurred while processing config file with OpenRouter: {str(e)}"

    logging.debug(f"OpenRouter: Using API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:]}")

    logging.debug(f"OpenRouter: Using Model: {openrouter_model}")

    if isinstance(input_data, str) and os.path.isfile(input_data):
        logging.debug("OpenRouter: Loading json data for summarization")
        with open(input_data, 'r') as file:
            data = json.load(file)
    else:
        logging.debug("OpenRouter: Using provided string data for summarization")
        data = input_data

    # DEBUG - Debug logging to identify sent data
    logging.debug(f"OpenRouter: Loaded data: {data[:500]}...(snipped to first 500 chars)")
    logging.debug(f"OpenRouter: Type of data: {type(data)}")

    if isinstance(data, dict) and 'summary' in data:
        # If the loaded data is a dictionary and already contains a summary, return it
        logging.debug("OpenRouter: Summary already exists in the loaded data")
        return data['summary']

    # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
    if isinstance(data, list):
        segments = data
        text = extract_text_from_segments(segments)
    elif isinstance(data, str):
        text = data
    else:
        raise ValueError("OpenRouter: Invalid input data format")

    openrouter_prompt = f"{input_data} \n\n\n\n{custom_prompt_arg}"
    logging.debug(f"openrouter: User Prompt being sent is {openrouter_prompt}")

    if temp is None:
        temp = 0.1
    temp = float(temp)
    if system_message is None:
        system_message = "You are a helpful AI assistant who does whatever the user requests."

    try:
        logging.debug("OpenRouter: Submitting request to API endpoint")
        print("OpenRouter: Submitting request to API endpoint")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
            },
            data=json.dumps({
                "model": openrouter_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": openrouter_prompt}
                ],
                "temperature": temp
            })
        )

        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("openrouter: Chat request successful")
                print("openrouter: Chat request successful.")
                return summary
            else:
                logging.error("openrouter: Expected data not found in API response.")
                return "openrouter: Expected data not found in API response."
        else:
            logging.error(f"openrouter:  API request failed with status code {response.status_code}: {response.text}")
            return f"openrouter: API request failed: {response.text}"
    except Exception as e:
        logging.error("openrouter: Error in processing: %s", str(e))
        return f"openrouter: Error occurred while processing chat request with openrouter: {str(e)}"


# FIXME: This function is not yet implemented properly
def chat_with_huggingface(api_key, input_data, custom_prompt_arg, system_prompt=None):
    loaded_config_data = load_and_log_configs()
    global huggingface_api_key
    logging.debug(f"huggingface: Summarization process starting...")
    try:
        # API key validation
        if not api_key:
            logging.info("HuggingFace: API key not provided as parameter")
            logging.info("HuggingFace: Attempting to use API key from config file")
            huggingface_api_key = loaded_config_data['api_keys']['openai']
        if not api_key or api_key.strip() == "":
            logging.error("HuggingFace: API key not found or is empty")
            return "HuggingFace: API Key Not Provided/Found in Config file or is empty"
        logging.debug(f"HuggingFace: Using API Key: {api_key[:5]}...{api_key[-5:]}")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        # Setup model
        huggingface_model = loaded_config_data['services']['huggingface']

        API_URL = f"https://api-inference.huggingface.co/models/{huggingface_model}"
        if system_prompt is not None:
            logging.debug("HuggingFace: Using provided system prompt")
            pass
        else:
            system_prompt = "You are a helpful assistant"

        huggingface_prompt = f"{input_data}\n\n\n\n{custom_prompt_arg}"
        logging.debug("huggingface: Prompt being sent is {huggingface_prompt}")
        data = {
            "inputs": f"{input_data}",
            "parameters": {"max_length": 8192, "min_length": 100}  # You can adjust max_length and min_length as needed
        }
        logging.debug("huggingface: Submitting request...")

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()[0]['generated_text'].strip()
            logging.debug("huggingface: Chat request successful")
            print("Chat request successful.")
            return summary
        else:
            logging.error(f"huggingface: Chat request failed with status code {response.status_code}: {response.text}")
            return f"Failed to process chat request, status code {response.status_code}: {response.text}"
    except Exception as e:
        logging.error("huggingface: Error in processing: %s", str(e))
        print(f"Error occurred while processing chat request with huggingface: {str(e)}")
        return None


def chat_with_deepseek(api_key, input_data, custom_prompt_arg, temp=None, system_message=None):
    logging.debug("DeepSeek: Summarization process starting...")
    try:
        logging.debug("DeepSeek: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            deepseek_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                deepseek_api_key = api_key
                logging.info("DeepSeek: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                deepseek_api_key = loaded_config_data['api_keys'].get('deepseek')
                if deepseek_api_key:
                    logging.info("DeepSeek: Using API key from config file")
                else:
                    logging.warning("DeepSeek: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not deepseek_api_key or not deepseek_api_key.strip():
            logging.error("DeepSeek: No valid API key available")
            # You might want to raise an exception here or handle this case as appropriate for your application
            # For example: raise ValueError("No valid deepseek API key available")


        logging.debug(f"DeepSeek: Using API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:]}")

        # Input data handling
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("DeepSeek: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("DeepSeek: Using provided string data for summarization")
            data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"DeepSeek: Loaded data: {data[:500]}...(snipped to first 500 chars)")
        logging.debug(f"DeepSeek: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("DeepSeek: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("DeepSeek: Invalid input data format")

        deepseek_model = loaded_config_data['services']['deepseek'] or "deepseek-chat"

        if temp is None:
            temp = 0.1
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"Deepseek API Key: {api_key[:5]}...{api_key[-5:] if api_key else None}")
        logging.debug("DeepSeek: Preparing data + prompt for submittal")
        deepseek_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "model": deepseek_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": deepseek_prompt}
            ],
            "stream": False,
            "temperature": temp
        }

        logging.debug("DeepSeek: Posting request")
        response = requests.post('https://api.deepseek.com/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("DeepSeek: Chat request successful")
                return summary
            else:
                logging.warning("DeepSeek: Chat response not found in the response data")
                return "DeepSeek: Chat response not available"
        else:
            logging.error(f"DeepSeek: Chat request failed with status code {response.status_code}")
            logging.error(f"DeepSeek: Error response: {response.text}")
            return f"DeepSeek: Failed to chat request summary. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"DeepSeek: Error in processing: {str(e)}", exc_info=True)
        return f"DeepSeek: Error occurred while processing chat request: {str(e)}"


def chat_with_mistral(api_key, input_data, custom_prompt_arg, temp=None, system_message=None):
    logging.debug("Mistral: Chat request made")
    try:
        logging.debug("Mistral: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            mistral_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                mistral_api_key = api_key
                logging.info("Mistral: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                mistral_api_key = loaded_config_data['api_keys'].get('mistral')
                if mistral_api_key:
                    logging.info("Mistral: Using API key from config file")
                else:
                    logging.warning("Mistral: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not mistral_api_key or not mistral_api_key.strip():
            logging.error("Mistral: No valid API key available")
            return "Mistral: No valid API key available"

        logging.debug(f"Mistral: Using API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:]}")

        logging.debug("Mistral: Using provided string data")
        data = input_data

        # Text extraction
        if isinstance(input_data, list):
            text = extract_text_from_segments(input_data)
        elif isinstance(input_data, str):
            text = input_data
        else:
            raise ValueError("Mistral: Invalid input data format")

        mistral_model = loaded_config_data['services'].get('mistral', "mistral-large-latest")

        temp = float(temp) if temp is not None else 0.2
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {mistral_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"Deepseek API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")
        logging.debug("Mistral: Preparing data + prompt for submittal")
        mistral_prompt = f"{custom_prompt_arg}\n\n\n\n{text} "
        data = {
            "model": mistral_model,
            "messages": [
                {"role": "system",
                 "content": system_message},
                {"role": "user",
                "content": mistral_prompt}
            ],
            "temperature": temp,
            "top_p": 1,
            "max_tokens": 4096,
            "stream": False,
            "safe_prompt": False
        }

        logging.debug("Mistral: Posting request")
        response = requests.post('https://api.mistral.ai/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("Mistral: request successful")
                return summary
            else:
                logging.warning("Mistral: Chat response not found in the response data")
                return "Mistral: Chat response not available"
        else:
            logging.error(f"Mistral: Chat request failed with status code {response.status_code}")
            logging.error(f"Mistral: Error response: {response.text}")
            return f"Mistral: Failed to process summary. Status code: {response.status_code}. Error: {response.text}"
    except Exception as e:
        logging.error(f"Mistral: Error in processing: {str(e)}", exc_info=True)
        return f"Mistral: Error occurred while processing Chat: {str(e)}"



# Stashed in here since OpenAI usage.... #FIXME
# FIXME - https://docs.vllm.ai/en/latest/getting_started/quickstart.html .... Great docs.
# def chat_with_vllm(input_data, custom_prompt_input, api_key=None, vllm_api_url="http://127.0.0.1:8000/v1/chat/completions", system_prompt=None):
#     loaded_config_data = load_and_log_configs()
#     llm_model = loaded_config_data['services']['vllm']
#     # API key validation
#     if api_key is None:
#         logging.info("vLLM: API key not provided as parameter")
#         logging.info("vLLM: Attempting to use API key from config file")
#         api_key = loaded_config_data['api_keys']['llama']
#
#     if api_key is None or api_key.strip() == "":
#         logging.info("vLLM: API key not found or is empty")
#     vllm_client = OpenAI(
#         base_url=vllm_api_url,
#         api_key=custom_prompt_input
#     )
#
#     if isinstance(input_data, str) and os.path.isfile(input_data):
#         logging.debug("vLLM: Loading json data for summarization")
#         with open(input_data, 'r') as file:
#             data = json.load(file)
#     else:
#         logging.debug("vLLM: Using provided string data for summarization")
#         data = input_data
#
#     logging.debug(f"vLLM: Loaded data: {data}")
#     logging.debug(f"vLLM: Type of data: {type(data)}")
#
#     if isinstance(data, dict) and 'summary' in data:
#         # If the loaded data is a dictionary and already contains a summary, return it
#         logging.debug("vLLM: Summary already exists in the loaded data")
#         return data['summary']
#
#     # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
#     if isinstance(data, list):
#         segments = data
#         text = extract_text_from_segments(segments)
#     elif isinstance(data, str):
#         text = data
#     else:
#         raise ValueError("Invalid input data format")
#
#
#     custom_prompt = custom_prompt_input
#
#     completion = client.chat.completions.create(
#         model=llm_model,
#         messages=[
#             {"role": "system", "content": f"{system_prompt}"},
#             {"role": "user", "content": f"{text} \n\n\n\n{custom_prompt}"}
#         ]
#     )
#     vllm_summary = completion.choices[0].message.content
#     return vllm_summary



#
#
#######################################################################################################################
