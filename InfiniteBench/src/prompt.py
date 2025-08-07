gpt4_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n{input}",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    # "longbook_sum_eng": "Summarize the book below:\n\n{context}",  # noqa
    "longbook_qa_eng": "Read the book below and answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe very concise.",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",  # noqa
    "longbook_sum_eng": "Summarize the following book.\n\n{context}",  # noqa
    "longbook_qa_chn": "请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{question}\n请尽量简短地回答。",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Compute the intermediate values in the following long expression.\n\n{context}",  # noqa
    "code_run": "Following is a set of Python functions. There is a function called named {func}.\n\n{context}\n\nPlease give me the exact number of the return value of {func_call}. Be concise. Your response must end with the final returned value.",  # noqa
    "code_debug": "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D.",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.",  # noqa
}

yarn_mistral_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",  # noqa
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",  # noqa
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",  # noqa
}

claude2_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n{input}\nThe pass key is",
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n{input}\nThe sequence of digits is",  # noqa
    "kv_retrieval": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n{input}",
    "longbook_sum_eng": "Summarize the following book.\n\n{context}",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",  # noqa
    "longbook_qa_eng": "Read the novel below and answer a question:\n\n{context}\n\n{input}\nPlease answer as short as possible. The answer is: ",  # noqa
    "longbook_qa_chn": "请根据以下书籍回答我的问题。\n\n{context}\n\n问题：{question}\n请尽量简短地回答。",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "In the file functions_module.py, there is a function called ${func}.\n\n\nHere is the content of functions_module.py:\n{context}\n\nPlease give me the exact number of the return value of {func_call}. Your response should end with the sentence \'The return value is:\'.",  # noqa
    "code_debug": "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect through the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D.",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the name used by the scriptwriter (before the colon marks) of one single character and nothing else.",  # noqa
}

kimi_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n{input}\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n{input}\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n{input}",  # noqa
    "longbook_sum_eng": "Summarize the book below:\n\n{file:{context}}",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\nQuestion: {question}\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}" + "{file:{document}}",  # noqa
    "longbook_qa_eng": "Read the book below and answer a question.\n\nQuestion: {question}\n\nBe very concise." + "{file:{context}}",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n问题：{question}\n答案：" + "{file:{context}}",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "In the file functions_module.py, there is a function called ${func}.\n\n\nHere is the content of functions_module.py:\n\nPlease give me the exact number of the return value of ${func_call}. Your response should end with the sentence 'The return value is:'." + "{context}",  # noqa
    "code_debug": "Below is a code repository where there is one single function with bugs that causes an error. Please tell me the name of that function.\nWhich function has bugs? Give me the final answer in this format: \"[FINAL ANSWER: XXX]\". Don't say anything else." + "{fcontext}",  # noqa
    # "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\nThe name that has been replaced with $$MASK$$ is likely" + "{context}",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is. Give me the answer using the name before the colons, don't say anything else.\n\n{context}",  # noqa
}

llama3_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",  # noqa
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",  # noqa
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",  # noqa
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",  # noqa
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",  # noqa
}

