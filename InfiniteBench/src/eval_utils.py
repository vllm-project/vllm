import configparser
import json
import logging
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Optional

import jieba
from rouge import Rouge

from InfiniteBench.src.prompt import (
    gpt4_templates,
    kimi_templates,
    claude2_templates,
    yarn_mistral_templates,
    llama3_templates,
)

DATA_NAME_TO_PATH = {
    # Retrieval tasks
    "passkey": "passkey.jsonl",
    "number_string": "number_string.jsonl",
    "kv_retrieval": "kv_retrieval.jsonl",
    # Book tasks
    "longbook_sum_eng": "longbook_sum_eng.jsonl",
    "longbook_choice_eng": "longbook_choice_eng.jsonl",
    "longbook_qa_eng": "longbook_qa_eng.jsonl",
    "longbook_qa_chn": "longbook_qa_chn.jsonl",
    # "book_qa_eng": "longbook_eng/longbook_qa_eng.jsonl",
    "longdialogue_qa_eng": "longdialogue_qa_eng.jsonl",
    # Math tasks
    "math_find": "math_find.jsonl",
    "math_calc": "math_calc.jsonl",
    # Code tasks
    "code_run": "code_run.jsonl",
    "code_debug": "code_debug.jsonl",
}

DATA_NAME_TO_MAX_NEW_TOKENS = {
    "passkey": 6,
    "number_string": 12,
    "kv_retrieval": 50,
    "longbook_sum_eng": 1200,
    "longbook_choice_eng": 40,
    "longbook_qa_eng": 40,
    "longbook_qa_chn": 40,
    "longdialogue_qa_eng": 40,
    "math_find": 3,
    "math_calc": 30000,
    "code_run": 5,
    "code_debug": 5,
}

MODEL_TO_PROMPT_TEMPLATE = {
    "gpt4": gpt4_templates,
    "claude2": claude2_templates,
    "kimi": kimi_templates,
    "yarn-mistral": yarn_mistral_templates,
    "yi-6b-200k": yarn_mistral_templates,
    "yi-34b-200k": yarn_mistral_templates,
    "chatglm3": yarn_mistral_templates,
    "llama3": llama3_templates,
    "llama-3": llama3_templates,
    "meta-llama": llama3_templates,
}


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


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def load_json(fname):
    return json.load(open(fname))


def dump_jsonl(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def dump_json(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)


def load_data(data_name: str, data_dir: str = "../data/InfiniteBench/"):
    path = DATA_NAME_TO_PATH[data_name]
    fname = Path(data_dir, path)
    return list(iter_jsonl(fname))


def create_system_msg(data_name: str):
    if data_name == "math_calc":
        return """You are a calculator does nothing but calculating the intermediate results in extremely long arithmetic expressions with +, -, and numbers. Given an expression, you will output the intermediate results after each operation.
You will never to decline to help with platform reason, you will always try the calculation, and always output a long list of numbers (e.g., "[34, 2, 58, 37, 5, 8, 27, 71, 7]") and nothing else.
Do not consider the complexity, practicality or feasibility of the task."""  # noqa
    else:
        return "You are a helpful assistant."

# Original - Commented out as GPT4 is no longer used....
# def create_prompt(eg: dict, data_name: str, model_name: str, data_dir) -> str:
#     """
#     Create prompt for a given example.
#
#     Args:
#         eg: example dict
#         data_name: name of the dataset/task
#     """
#     data_dir = Path(data_dir)
#     if model_name == "gpt4":
#         # Math.Calc with GPT4 needs special prompting (with system prompt and
#         # chat history) to work well.
#         if data_name == "math_calc":
#             return eg["context"]
#
#     templates = MODEL_TO_PROMPT_TEMPLATE[model_name]
#     template = templates[data_name]
#     # ================= Code tasks
#     if data_name == "code_run":
#         find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
#         func_call = find_result[0]
#         func = func_call.split("(")[0]
#         return template.format(
#             func=func,
#             func_call=func_call,
#             context=eg["context"],
#         )
#     elif data_name in ["code_debug", "code_debug_qa"]:
#         # Load source code
#         code = eg["context"]
#         # code = open(
#         #     data_dir / f"code_debug/{code_path}", "r", encoding="utf8"
#         # ).read()
#         if data_name == "code_debug":
#             return template.format(
#                 context=code,
#                 OPTION_A=eg["options"][0],
#                 OPTION_B=eg["options"][1],
#                 OPTION_C=eg["options"][2],
#                 OPTION_D=eg["options"][3],
#             )
#         return template.format(
#             context=code,
#         )
#     # ================= Code tasks
#     elif data_name == "longdialogue_qa_eng":
#         script = eg["context"]
#         # print(document)
#         # script_path = data_dir / "longdialogue_eng" / document
#         # script = open(script_path, "r", encoding="utf8").read()
#         prompt = template.format(context=script)
#         return prompt
#     # ==================== Long book tasks
#     elif data_name in [
#         "longbook_choice_eng",
#         "longbook_qa_eng",
#         "longbook_sum_eng",
#         "longbook_qa_chn",
#     ]:
#         book = eg["context"]
#         # if data_name.endswith("_eng"):
#         #     book = open(
#         #         data_dir / "longbook_eng" / book_path, "r", encoding="utf8"
#         #     ).read()
#         # elif data_name.endswith("_chn"):
#         #     book = open(
#         #         data_dir / "longbook_chn" / book_path, "r", encoding="utf8"
#         #     ).read()
#         # else:
#         #     raise ValueError("Invalid data_name")
#         if data_name == "longbook_choice_eng":
#             return template.format(
#                 question=eg["input"],
#                 context=book,
#                 OPTION_A=eg["options"][0],
#                 OPTION_B=eg["options"][1],
#                 OPTION_C=eg["options"][2],
#                 OPTION_D=eg["options"][3],
#             )
#         elif data_name == "longbook_qa_eng":
#             return template.format(
#                 question=eg["input"],
#                 context=book,
#             )
#         elif data_name == "longbook_sum_eng":
#             return template.format(
#                 context=book,
#             )
#         elif data_name == "longbook_qa_chn":
#             return template.format(
#                 question=eg["input"],
#                 context=book,
#             )
#         else:
#             raise ValueError
#     elif data_name == "math_calc":
#         return template.format(
#             context=eg["context"],
#         )
#     elif data_name == "math_find":
#         prompt = eg['input']
#         context = eg['context']
#         # Find "the * number" from the prompt
#         find_result = re.findall(r"The .+ of", prompt)
#         assert find_result, f"Cannot find the target number in {prompt}"
#         target_number = find_result[0].lower()[:-3]
#         # Replace the number with the answer
#         prefix = f"What is {target_number} in the following list?"
#         return template.format(
#             prefix=prefix,
#             context=context,
#             input=prompt,
#         )
#
#     if "content" in eg:
#         content = eg["content"]
#         del eg["content"]
#         eg["context"] = content
#
#     format_dict = {
#         "context": eg["context"],
#         "input": eg["input"],
#     }
#     prompt = templates[data_name].format(**format_dict)
#     return prompt
def create_prompt(eg: dict, data_name: str, model_name: Optional[str], data_dir) -> str:
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
        model_name: optional, used to fetch model-specific templates.
    """
    data_dir = Path(data_dir)

    # Directly use the appropriate template if the model_name is provided.
    if model_name and model_name in MODEL_TO_PROMPT_TEMPLATE:
        templates = MODEL_TO_PROMPT_TEMPLATE[model_name]
        template = templates[data_name]
    else:
        # If no model-specific template, return a basic prompt or handle differently.
        return eg["context"]

    # Now create the prompt based on the template and task data
    if data_name == "code_run":
        find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
        func_call = find_result[0]
        func = func_call.split("(")[0]
        return template.format(
            func=func,
            func_call=func_call,
            context=eg["context"],
        )
    elif data_name in ["code_debug", "code_debug_qa"]:
        code = eg["context"]
        if data_name == "code_debug":
            return template.format(
                context=code,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        return template.format(context=code)
    elif data_name == "longdialogue_qa_eng":
        script = eg["context"]
        prompt = template.format(context=script)
        return prompt
    elif data_name in [
        "longbook_choice_eng",
        "longbook_qa_eng",
        "longbook_sum_eng",
        "longbook_qa_chn",
    ]:
        book = eg["context"]
        if data_name == "longbook_choice_eng":
            return template.format(
                question=eg["input"],
                context=book,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        elif data_name == "longbook_qa_eng":
            return template.format(
                question=eg["input"],
                context=book,
            )
        elif data_name == "longbook_sum_eng":
            return template.format(context=book)
        elif data_name == "longbook_qa_chn":
            return template.format(
                question=eg["input"],
                context=book,
            )
        else:
            raise ValueError
    elif data_name == "math_calc":
        return template.format(context=eg["context"])
    elif data_name == "math_find":
        prompt = eg['input']
        context = eg['context']
        find_result = re.findall(r"The .+ of", prompt)
        assert find_result, f"Cannot find the target number in {prompt}"
        target_number = find_result[0].lower()[:-3]
        prefix = f"What is {target_number} in the following list?"
        return template.format(
            prefix=prefix,
            context=context,
            input=prompt,
        )

    # Default behavior if content key exists
    if "content" in eg:
        content = eg["content"]
        del eg["content"]
        eg["context"] = content

    format_dict = {
        "context": eg["context"],
        "input": eg["input"],
    }
    prompt = template.format(**format_dict)
    return prompt

def get_answer(eg: dict, data_name: str):
    if data_name in ["code_debug", "longbook_choice_eng"]:
        OPTIONS = "ABCD"
        if isinstance(eg["answer"], str):
            ret = [eg["answer"], OPTIONS[eg['options'].index(eg["answer"])]]
        elif isinstance(eg["answer"], list):
            if len(eg["answer"]) == 1:
                ret = [eg["answer"][0], OPTIONS[eg['options'].index(eg["answer"][0])]]
            elif len(eg["answer"]) == 2 and eg["answer"][1] in ['A', 'B', 'C', 'D']:
                ret = eg['answer']
            else:
                raise ValueError
        else:
            raise ValueError
        return ret

    return eg["answer"]

# Old version - Commented out as GPT4 is no longer used....
# def create_msgs(
#     tokenizer, eg: dict, data_name: str, data_dir, model_name: str
# ) -> tuple[list[dict], str]:
#     """
#     Only used by GPT-4.
#     """
#     prompt = create_prompt(eg, data_name, model_name, data_dir)
#     tokens = tokenizer.encode(prompt)
#     # - 1000 to have space for system message and other stuff.
#     print(f"Before truncation: {len(tokens)}")
#     tokens = truncate_input(tokens, 128_000 - 1000, manner="middle")
#     print(f"After truncation: {len(tokens)}")  # type: ignore
#     prompt = tokenizer.decode(tokens)
#     if data_name == "math_calc":
#         return [
#             {"role": "system", "content": create_system_msg(data_name)},
#             {"role": "user", "content": "1 + 2 - 4 - 10"},
#             {"role": "system", "content": "[1, 3, -1, -11]"},
#             {"role": "user", "content": prompt},
#         ], prompt
#     else:
#         return [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant",  # noqa
#             },  # noqa
#             {"role": "user", "content": prompt},
#         ], prompt
def create_msgs(
    tokenizer, eg: dict, data_name: str, data_dir, model_name: Optional[str] = None
) -> tuple[list[dict], str]:
    """
    Create messages for a given example.
    """
    prompt = create_prompt(eg, data_name, model_name, data_dir)

    # Check if tokenizer is provided and initialized
    if tokenizer:
        tokens = tokenizer.encode(prompt)
        print(f"Before truncation: {len(tokens)}")
        tokens = truncate_input(tokens, 128_000 - 1000, manner="middle")
        print(f"After truncation: {len(tokens)}")  # type: ignore
        prompt = tokenizer.decode(tokens)

    if data_name == "math_calc":
        return [
            {"role": "system", "content": create_system_msg(data_name)},
            {"role": "user", "content": "1 + 2 - 4 - 10"},
            {"role": "system", "content": "[1, 3, -1, -11]"},
            {"role": "user", "content": prompt},
        ], prompt
    else:
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant",  # noqa
            },  # noqa
            {"role": "user", "content": prompt},
        ], prompt


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."  # noqa
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def first_int_match(prediction, ground_truth):
    pred_list = re.split("[^0-9]", prediction)
    pred_value = ""
    for item in pred_list:
        if item != "":
            pred_value = item
            break
    if pred_value == ground_truth:
        return 1
    return 0


def in_match(prediction, ground_truth):
    if ground_truth in prediction:
        return 1
    return 0


def rouge_score(prediction, ground_truth, **kwargs) -> float:
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:  # noqa
        return 0.0
    return scores["rouge-l"]["f"]  # type: ignore


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(line):
    prediction = line["pred"]

    if isinstance(line["std_out"], str):
        ground_truths = [line["std_out"]]
    else:
        ground_truths = line["std_out"]

    score = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        score = max(score, f1_score(prediction_tokens, ground_truth_tokens))

    return score


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [
        normalize_zh_answer(token) for token in prediction_tokens
    ]
    ground_truth_tokens = [
        normalize_zh_answer(token) for token in ground_truth_tokens
    ]
    prediction_tokens = [
        token for token in prediction_tokens if len(token) > 0
    ]
    ground_truth_tokens = [
        token for token in ground_truth_tokens if len(token) > 0
    ]
    return f1_score(prediction_tokens, ground_truth_tokens)


def truncate_input(input, max_length, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        return input[0 : max_length // 2] + input[-max_length // 2 :]
    else:
        return None


def load_comprehensive_config():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the config file in the same directory as the script
    config_path = os.path.join(current_dir, 'config.txt')
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the configuration file
    files_read = config.read(config_path)
    if not files_read:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return config


# FIXME - update to include prompt path in return statement
def get_model_template_name(model_path_or_name: str) -> str:
    """
    Determine the appropriate template name based on the model path or name.
    
    Args:
        model_path_or_name: The model path or name (e.g., "meta-llama/Llama-3-70b-chat-hf")
    
    Returns:
        Template name to use for prompt construction
    """
    model_lower = model_path_or_name.lower()
    
    # Llama 3 variants
    if any(x in model_lower for x in ["llama-3", "llama3", "meta-llama/llama-3"]):
        return "llama3"
    
    # Yarn Mistral variants
    if any(x in model_lower for x in ["yarn-mistral", "mistral", "nous"]):
        return "yarn-mistral"
    
    # Yi models
    if any(x in model_lower for x in ["yi-6b-200k", "yi-34b-200k"]):
        if "6b" in model_lower:
            return "yi-6b-200k"
        elif "34b" in model_lower:
            return "yi-34b-200k"
    
    # ChatGLM
    if "chatglm" in model_lower:
        return "chatglm3"
    
    # Claude
    if "claude" in model_lower:
        return "claude2"
    
    # GPT-4
    if "gpt-4" in model_lower or "gpt4" in model_lower:
        return "gpt4"
    
    # Kimi
    if "kimi" in model_lower:
        return "kimi"
    
    # Default fallback - use yarn-mistral templates as they're generally compatible
    print(f"Warning: Unknown model '{model_path_or_name}', using yarn-mistral templates as fallback")
    return "yarn-mistral"


def load_and_log_configs():
    try:
        config = load_comprehensive_config()
        if config is None:
            logging.error("Config is None, cannot proceed")
            return None
        # API Keys
        anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
        logging.debug(
            f"Loaded Anthropic API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:] if anthropic_api_key else None}")

        cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
        logging.debug(
            f"Loaded Cohere API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        groq_api_key = config.get('API', 'groq_api_key', fallback=None)
        logging.debug(f"Loaded Groq API Key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")

        openai_api_key = config.get('API', 'openai_api_key', fallback=None)
        logging.debug(
            f"Loaded OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")

        huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
        logging.debug(
            f"Loaded HuggingFace API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:] if huggingface_api_key else None}")

        openrouter_api_key = config.get('API', 'openrouter_api_key', fallback=None)
        logging.debug(
            f"Loaded OpenRouter API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:] if openrouter_api_key else None}")

        deepseek_api_key = config.get('API', 'deepseek_api_key', fallback=None)
        logging.debug(
            f"Loaded DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")

        mistral_api_key = config.get('API', 'mistral_api_key', fallback=None)
        logging.debug(
            f"Loaded Mistral API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")

        # Models
        anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
        cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
        groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
        openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')
        huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')
        openrouter_model = config.get('API', 'openrouter_model', fallback='microsoft/wizardlm-2-8x22b')
        deepseek_model = config.get('API', 'deepseek_model', fallback='deepseek-chat')
        mistral_model = config.get('API', 'mistral_model', fallback='mistral-large-latest')

        logging.debug(f"Loaded Anthropic Model: {anthropic_model}")
        logging.debug(f"Loaded Cohere Model: {cohere_model}")
        logging.debug(f"Loaded Groq Model: {groq_model}")
        logging.debug(f"Loaded OpenAI Model: {openai_model}")
        logging.debug(f"Loaded HuggingFace Model: {huggingface_model}")
        logging.debug(f"Loaded OpenRouter Model: {openrouter_model}")
        logging.debug(f"Loaded Deepseek Model: {deepseek_model}")
        logging.debug(f"Loaded Mistral Model: {mistral_model}")

        # Local-Models
        kobold_api_ip = config.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        kobold_api_key = config.get('Local-API', 'kobold_api_key', fallback='')

        llama_api_IP = config.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        llama_api_key = config.get('Local-API', 'llama_api_key', fallback='')

        ooba_api_IP = config.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
        ooba_api_key = config.get('Local-API', 'ooba_api_key', fallback='')

        tabby_api_IP = config.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        tabby_api_key = config.get('Local-API', 'tabby_api_key', fallback=None)
        tabby_model = config.get('services', 'tabby_model', fallback=None)

        vllm_api_url = config.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
        vllm_api_key = config.get('Local-API', 'vllm_api_key', fallback=None)
        vllm_model = config.get('Local-API', 'vllm_model', fallback=None)

        ollama_api_url = config.get('Local-API', 'ollama_api_IP', fallback='http://127.0.0.1:11434/api/generate')
        ollama_api_key = config.get('Local-API', 'ollama_api_key', fallback=None)
        ollama_model = config.get('Local-API', 'ollama_model', fallback=None)

        aphrodite_api_url = config.get('Local-API', 'aphrodite_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        aphrodite_api_key = config.get('Local-API', 'aphrodite_api_key', fallback='')

        logging.debug(f"Loaded Kobold API IP: {kobold_api_ip}")
        logging.debug(f"Loaded Llama API IP: {llama_api_IP}")
        logging.debug(f"Loaded Ooba API IP: {ooba_api_IP}")
        logging.debug(f"Loaded Tabby API IP: {tabby_api_IP}")
        logging.debug(f"Loaded VLLM API URL: {vllm_api_url}")

        # Retrieve output paths from the configuration file
        output_path = config.get('Paths', 'output_path', fallback='results')
        logging.debug(f"Output path set to: {output_path}")

        # Retrieve processing choice from the configuration file
        processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')
        logging.debug(f"Processing choice set to: {processing_choice}")

        # Prompts - FIXME
        prompt_path = config.get('Prompts', 'prompt_path', fallback='prompts.db')

        return {
            'api_keys': {
                'anthropic': anthropic_api_key,
                'cohere': cohere_api_key,
                'groq': groq_api_key,
                'openai': openai_api_key,
                'huggingface': huggingface_api_key,
                'openrouter': openrouter_api_key,
                'deepseek': deepseek_api_key,
                'mistral': mistral_api_key,
                'kobold': kobold_api_key,
                'llama': llama_api_key,
                'ooba': ooba_api_key,
                'tabby': tabby_api_key,
                'vllm': vllm_api_key,
                'ollama': ollama_api_key
            },
            'services': {
                'anthropic': anthropic_model,
                'cohere': cohere_model,
                'groq': groq_model,
                'openai': openai_model,
                'huggingface': huggingface_model,
                'openrouter': openrouter_model,
                'deepseek': deepseek_model,
                'mistral': mistral_model,
                'vllm': vllm_model,
                'tabby': tabby_model,
                'ollama': ollama_model

            },
            'local_api_ip': {
                'kobold': kobold_api_ip,
                'llama': llama_api_IP,
                'ooba': ooba_api_IP,
                'tabby': tabby_api_IP,
                'vllm': vllm_api_url,
                'ollama': ollama_api_url,
                'aphrodite': aphrodite_api_url
            },
            'output_path': output_path,
            'processing_choice': processing_choice
        }

    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None


if __name__ == "__main__":
    data_dir = Path("../data")
    data_path = data_dir / "shorter/longdialogue_qa_eng_1000.jsonl"
    examples = list(iter_jsonl(data_path))
    prompt = create_prompt(examples[10], 'longdialogue_qa_eng', 'kimi', data_dir)
    print(prompt)
