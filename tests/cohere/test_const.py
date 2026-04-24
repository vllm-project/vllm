# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
import json

C3_grammar = """\n        ws    ::= (" " | "\\t" | "\\n")*\n        root ::= ws "[" ws tool ws ("," ws tool)*  ws "]" ws              "<|END_ACTION|>" ws\n        tool ::= add_numbers | multiply_numbers\n        add_numbers ::= add_numbersroot\nadd_numbersbasic_number ::= ((add_numbersadd_numbersbasic_number_7 add_numbersadd_numbersbasic_number_3 add_numbersadd_numbersbasic_number_6))\nadd_numbersadd_numbersroot_prop_0 ::= (("\\"" add_numbersadd_numbersadd_numbersroot_prop_0_1 "\\"")) (=(add_numbersadd_numbersroot_part_0))\nadd_numbersadd_numbersroot_prop_1 ::= (("\\"add_numbers\\""))\nadd_numbersadd_numbersroot_prop_2 ::= (("{" [ \\n\\t]* "\\"num1\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersbasic_number [ \\n\\t]* "," [ \\n\\t]* "\\"num2\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersbasic_number [ \\n\\t]* "}"))\nadd_numbersadd_numbersroot_part_1 ::= ("" | ([ \\n\\t]* "," [ \\n\\t]* "\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_2))\nadd_numbersadd_numbersroot_part_0 ::= ((add_numbersadd_numbersroot_part_1) | ([ \\n\\t]* "," [ \\n\\t]* "\\"tool_name\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_1 add_numbersadd_numbersroot_part_1))\nadd_numbersroot ::= (("{" [ \\n\\t]* add_numbersadd_numbersroot_1 [ \\n\\t]* "}") | ("{" [ \\n\\t]* "}"))\nadd_numbersadd_numbersbasic_number_1 ::= ("" | ("-")) (=([1-9] [0-9]*))\nadd_numbersadd_numbersbasic_number_2 ::= (([0-9] add_numbersadd_numbersbasic_number_2) | ([0-9]))\nadd_numbersadd_numbersbasic_number_3 ::= ("" | ("." add_numbersadd_numbersbasic_number_2)) (=(add_numbersadd_numbersbasic_number_6))\nadd_numbersadd_numbersbasic_number_4 ::= ("" | ([+\\-])) (=(add_numbersadd_numbersbasic_number_5))\nadd_numbersadd_numbersbasic_number_5 ::= (([0-9] add_numbersadd_numbersbasic_number_5) | ([0-9]))\nadd_numbersadd_numbersbasic_number_6 ::= ("" | ([eE] add_numbersadd_numbersbasic_number_4 add_numbersadd_numbersbasic_number_5))\nadd_numbersadd_numbersadd_numbersroot_prop_0_1 ::= (([0-9] add_numbersadd_numbersadd_numbersroot_prop_0_1) | ([0-9])) (=("\\""))\nadd_numbersadd_numbersbasic_number_7 ::= (("0") | (add_numbersadd_numbersbasic_number_1 [1-9] [0-9]*)) (=(add_numbersadd_numbersbasic_number_3 add_numbersadd_numbersbasic_number_6))\nadd_numbersadd_numbersroot_1 ::= (("\\"tool_call_id\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_0 add_numbersadd_numbersroot_part_0) | ("\\"tool_name\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_1 add_numbersadd_numbersroot_part_1) | ("\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_2)) (=([ \\n\\t]* "}"))\n\n    multiply_numbers ::= multiply_numbersroot\nmultiply_numbersbasic_escape ::= (([\\"\\\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(multiply_numbersmultiply_numbersbasic_string_sub))\nmultiply_numbersmultiply_numbersbasic_string_sub ::= (("\\"") | ([^\\0-\\x1f\\"\\\\\\r\\n] multiply_numbersmultiply_numbersbasic_string_sub) | ("\\\\" multiply_numbersbasic_escape multiply_numbersmultiply_numbersbasic_string_sub)) (=([ \\n\\t]* [,}\\]:]))\nmultiply_numbersbasic_number ::= ((multiply_numbersmultiply_numbersbasic_number_7 multiply_numbersmultiply_numbersbasic_number_3 multiply_numbersmultiply_numbersbasic_number_6)) (=([ \\n\\t]* "}"))\nmultiply_numbersbasic_string ::= (("\\"" multiply_numbersmultiply_numbersbasic_string_sub)) (=([ \\n\\t]* "," [ \\n\\t]* "\\"num2\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersbasic_number [ \\n\\t]* "}"))\nmultiply_numbersmultiply_numbersroot_prop_0 ::= (("\\"" multiply_numbersmultiply_numbersmultiply_numbersroot_prop_0_1 "\\"")) (=(multiply_numbersmultiply_numbersroot_part_0))\nmultiply_numbersmultiply_numbersroot_prop_1 ::= (("\\"multiply_numbers\\""))\nmultiply_numbersmultiply_numbersroot_prop_2 ::= (("{" [ \\n\\t]* "\\"num1\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersbasic_string [ \\n\\t]* "," [ \\n\\t]* "\\"num2\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersbasic_number [ \\n\\t]* "}"))\nmultiply_numbersmultiply_numbersroot_part_1 ::= ("" | ([ \\n\\t]* "," [ \\n\\t]* "\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_2))\nmultiply_numbersmultiply_numbersroot_part_0 ::= ((multiply_numbersmultiply_numbersroot_part_1) | ([ \\n\\t]* "," [ \\n\\t]* "\\"tool_name\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_1 multiply_numbersmultiply_numbersroot_part_1))\nmultiply_numbersroot ::= (("{" [ \\n\\t]* multiply_numbersmultiply_numbersroot_1 [ \\n\\t]* "}") | ("{" [ \\n\\t]* "}"))\nmultiply_numbersmultiply_numbersbasic_number_1 ::= ("" | ("-")) (=([1-9] [0-9]*))\nmultiply_numbersmultiply_numbersbasic_number_2 ::= (([0-9] multiply_numbersmultiply_numbersbasic_number_2) | ([0-9]))\nmultiply_numbersmultiply_numbersbasic_number_3 ::= ("" | ("." multiply_numbersmultiply_numbersbasic_number_2)) (=(multiply_numbersmultiply_numbersbasic_number_6))\nmultiply_numbersmultiply_numbersbasic_number_4 ::= ("" | ([+\\-])) (=(multiply_numbersmultiply_numbersbasic_number_5))\nmultiply_numbersmultiply_numbersbasic_number_5 ::= (([0-9] multiply_numbersmultiply_numbersbasic_number_5) | ([0-9]))\nmultiply_numbersmultiply_numbersbasic_number_6 ::= ("" | ([eE] multiply_numbersmultiply_numbersbasic_number_4 multiply_numbersmultiply_numbersbasic_number_5))\nmultiply_numbersmultiply_numbersmultiply_numbersroot_prop_0_1 ::= (([0-9] multiply_numbersmultiply_numbersmultiply_numbersroot_prop_0_1) | ([0-9])) (=("\\""))\nmultiply_numbersmultiply_numbersbasic_number_7 ::= (("0") | (multiply_numbersmultiply_numbersbasic_number_1 [1-9] [0-9]*)) (=(multiply_numbersmultiply_numbersbasic_number_3 multiply_numbersmultiply_numbersbasic_number_6))\nmultiply_numbersmultiply_numbersroot_1 ::= (("\\"tool_call_id\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_0 multiply_numbersmultiply_numbersroot_part_0) | ("\\"tool_name\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_1 multiply_numbersmultiply_numbersroot_part_1) | ("\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_2)) (=([ \\n\\t]* "}"))\n\n        """

C2_grammar = """\n        root ::= ws "[" ws tool ws ("," ws tool)*  ws "]" ws              "```" ws\n        ws    ::= (" " | "\\t" | "\\n")*\n        tool ::= add_numbers | multiply_numbers\n        add_numbers ::= add_numbersroot\nadd_numbersbasic_number ::= ((add_numbersadd_numbersbasic_number_7 add_numbersadd_numbersbasic_number_3 add_numbersadd_numbersbasic_number_6))\nadd_numbersadd_numbersroot_prop_0 ::= (("\\"add_numbers\\"")) (=(add_numbersadd_numbersroot_part_0))\nadd_numbersadd_numbersroot_prop_1 ::= (("{" [ \\n\\t]* "\\"num1\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersbasic_number [ \\n\\t]* "," [ \\n\\t]* "\\"num2\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersbasic_number [ \\n\\t]* "}"))\nadd_numbersadd_numbersroot_part_0 ::= ("" | ([ \\n\\t]* "," [ \\n\\t]* "\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_1))\nadd_numbersroot ::= (("{" [ \\n\\t]* add_numbersadd_numbersroot_1 [ \\n\\t]* "}") | ("{" [ \\n\\t]* "}"))\nadd_numbersadd_numbersbasic_number_1 ::= ("" | ("-")) (=([1-9] [0-9]*))\nadd_numbersadd_numbersbasic_number_2 ::= (([0-9] add_numbersadd_numbersbasic_number_2) | ([0-9]))\nadd_numbersadd_numbersbasic_number_3 ::= ("" | ("." add_numbersadd_numbersbasic_number_2)) (=(add_numbersadd_numbersbasic_number_6))\nadd_numbersadd_numbersbasic_number_4 ::= ("" | ([+\\-])) (=(add_numbersadd_numbersbasic_number_5))\nadd_numbersadd_numbersbasic_number_5 ::= (([0-9] add_numbersadd_numbersbasic_number_5) | ([0-9]))\nadd_numbersadd_numbersbasic_number_6 ::= ("" | ([eE] add_numbersadd_numbersbasic_number_4 add_numbersadd_numbersbasic_number_5))\nadd_numbersadd_numbersbasic_number_7 ::= (("0") | (add_numbersadd_numbersbasic_number_1 [1-9] [0-9]*)) (=(add_numbersadd_numbersbasic_number_3 add_numbersadd_numbersbasic_number_6))\nadd_numbersadd_numbersroot_1 ::= (("\\"tool_name\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_0 add_numbersadd_numbersroot_part_0) | ("\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* add_numbersadd_numbersroot_prop_1)) (=([ \\n\\t]* "}"))\n\n    multiply_numbers ::= multiply_numbersroot\nmultiply_numbersbasic_escape ::= (([\\"\\\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9])) (=(multiply_numbersmultiply_numbersbasic_string_sub))\nmultiply_numbersmultiply_numbersbasic_string_sub ::= (("\\"") | ([^\\0-\\x1f\\"\\\\\\r\\n] multiply_numbersmultiply_numbersbasic_string_sub) | ("\\\\" multiply_numbersbasic_escape multiply_numbersmultiply_numbersbasic_string_sub)) (=([ \\n\\t]* [,}\\]:]))\nmultiply_numbersbasic_number ::= ((multiply_numbersmultiply_numbersbasic_number_7 multiply_numbersmultiply_numbersbasic_number_3 multiply_numbersmultiply_numbersbasic_number_6)) (=([ \\n\\t]* "}"))\nmultiply_numbersbasic_string ::= (("\\"" multiply_numbersmultiply_numbersbasic_string_sub)) (=([ \\n\\t]* "," [ \\n\\t]* "\\"num2\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersbasic_number [ \\n\\t]* "}"))\nmultiply_numbersmultiply_numbersroot_prop_0 ::= (("\\"multiply_numbers\\"")) (=(multiply_numbersmultiply_numbersroot_part_0))\nmultiply_numbersmultiply_numbersroot_prop_1 ::= (("{" [ \\n\\t]* "\\"num1\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersbasic_string [ \\n\\t]* "," [ \\n\\t]* "\\"num2\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersbasic_number [ \\n\\t]* "}"))\nmultiply_numbersmultiply_numbersroot_part_0 ::= ("" | ([ \\n\\t]* "," [ \\n\\t]* "\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_1))\nmultiply_numbersroot ::= (("{" [ \\n\\t]* multiply_numbersmultiply_numbersroot_1 [ \\n\\t]* "}") | ("{" [ \\n\\t]* "}"))\nmultiply_numbersmultiply_numbersbasic_number_1 ::= ("" | ("-")) (=([1-9] [0-9]*))\nmultiply_numbersmultiply_numbersbasic_number_2 ::= (([0-9] multiply_numbersmultiply_numbersbasic_number_2) | ([0-9]))\nmultiply_numbersmultiply_numbersbasic_number_3 ::= ("" | ("." multiply_numbersmultiply_numbersbasic_number_2)) (=(multiply_numbersmultiply_numbersbasic_number_6))\nmultiply_numbersmultiply_numbersbasic_number_4 ::= ("" | ([+\\-])) (=(multiply_numbersmultiply_numbersbasic_number_5))\nmultiply_numbersmultiply_numbersbasic_number_5 ::= (([0-9] multiply_numbersmultiply_numbersbasic_number_5) | ([0-9]))\nmultiply_numbersmultiply_numbersbasic_number_6 ::= ("" | ([eE] multiply_numbersmultiply_numbersbasic_number_4 multiply_numbersmultiply_numbersbasic_number_5))\nmultiply_numbersmultiply_numbersbasic_number_7 ::= (("0") | (multiply_numbersmultiply_numbersbasic_number_1 [1-9] [0-9]*)) (=(multiply_numbersmultiply_numbersbasic_number_3 multiply_numbersmultiply_numbersbasic_number_6))\nmultiply_numbersmultiply_numbersroot_1 ::= (("\\"tool_name\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_0 multiply_numbersmultiply_numbersroot_part_0) | ("\\"parameters\\"" [ \\n\\t]* ":" [ \\n\\t]* multiply_numbersmultiply_numbersroot_prop_1)) (=([ \\n\\t]* "}"))\n\n        """

C3_tool_prompt_1 = r"""<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble\nYou are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you will not provide any content that could directly or indirectly lead to harmful
outcomes.\n\nYour information cutoff date is June 2024.\n\nYou have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi, Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.\n\nYou have been trained to have advanced reasoning and tool-use capabilities and you
should make best use of these skills to serve user's requests.\n\n## Tool Use\nThink about how you can make best use of the provided tools to help with the
task and come up with a high level plan that you will execute first.\n\n0. Start by writing <|START_THINKING|> followed by a detailed step by step plan of how you will solve the problem. For each step explain your thinking fully and give details of required tool calls (if needed). Unless specified otherwise, you write your plan in natural language. When you finish, close it out with <|END_THINKING|>.\n
You can optionally choose to skip this step when the user request is so straightforward to address that only a trivial plan would be needed.\n    NOTE: You MUST skip this step when you are directly responding to the user's request without using any tools.\n\nThen carry out your plan by
repeatedly executing the following steps.\n1. Action: write <|START_ACTION|> followed by a list of JSON-formatted tool calls, with each one containing \"tool_name\" and \"parameters\" fields.\n    When there are multiple tool calls which are completely independent of each other (i.e. they can be executed in parallel), you should list them out all together
in one step. When you finish, close it out with <|END_ACTION|>.\n2. Observation: you will then receive results of those tool calls in JSON format in the very next turn, wrapped around by <|START_TOOL_RESULT|> and <|END_TOOL_RESULT|>. Carefully observe those results and think about what to do next. Note that these results will be provided to you in a separate turn. NEVER hallucinate results.\n
 Every tool call produces a list of results (when a tool call produces no result or a single result, it'll still get wrapped inside a list). Each result is clearly linked to its originating tool call via its \"tool_call_id\".\n3. Reflection: start the next turn by writing <|START_THINKING|> followed by what you've figured out so far, any changes you need to make to your plan, and what you
 will do next. When you finish, close it out with <|END_THINKING|>.\n    You can optionally choose to skip this step when everything is going according to plan and no special pieces of information or reasoning chains need to be recorded.\n    NOTE: You MUST skip this step when you are done with tool-use actions and are ready to respond to the user.\n\nYou can repeat the above 3 steps multiple times
 (could be 0 times too if no suitable tool calls are available or needed), until you decide it's time to finally respond to the user.\n\n4. Response: then break out of the loop and write <|START_RESPONSE|> followed by a piece of text which serves as a response to the user's last request. Use all previous tool calls and results to help you when formulating your response. When you finish, close it out with <|END_RESPONSE|>.\n\n## Grounding\nImportantly, note that \"Reflection\" and \"Response\" above can be grounded.\nGrounding means you associate pieces of texts (called \"spans\") with those specific tool results that support them (called \"sources\"). And you use a pair of tags \"<co>\" and \"</co>\" to indicate when a span can be grounded onto a list of sources, listing them out in the closing tag. Sources from the
 same tool call are grouped together and listed as \"{tool_call_id}:[{list of result indices}]\", before they are joined together by \",\". E.g., \"<co>span</co: 0:[1,2],1:[0]>\" means that \"span\" is supported by result 1 and 2 from \"tool_call_id=0\" as well as result 0 from \"tool_call_id=1\".\n\n## Available Tools\nHere is the list of tools that you have available to you.\nYou can
 ONLY use the tools listed here. When a tool is not listed below, it is NOT available and you should NEVER attempt to use it.\nEach tool is represented as a JSON object with fields like \"name\", \"description\", \"parameters\" (per JSON Schema), and optionally, \"responses\" (per JSON Schema).\n\n```json\n[\n    {\"name\": \"add_numbers\", \"description\": \"adds two integers: num1, and num2\", \"parameters\": {\"type\": \"object\", \"properties\": {\"num1\": {\"type\": \"number\", \"description\": \"num1: first number\"}, \"num2\": {\"type\": \"number\", \"description\": \"num2: 2nd number\"}}, \"required
 \": [\"num1\", \"num2\"]}, \"responses\": null},\n    {\"name\": \"multiply_numbers\", \"description\": \"multiply two integers: num1, and num2\", \"parameters\": {\"type\": \"object\", \"properties\": {\"num1\": {\"type\": \"string\", \"description\": \"num1: first number\"}, \"num2\": {\"type\": \"number\", \"description\": \"num2: 2nd number\"}}, \"required\": [\"num1\", \"num2\"]}, \"responses\": null}\n]\n```\n\n# Default Preamble\nThe following instructions are your defaults unless specified elsewhere in developer preamble
 or user prompt.\n- Your name is Command.\n- You are a large language model built by Cohere.\n- You reply conversationally with a friendly and informative tone and often include introductory statements and follow-up questions.\n- If the input is ambiguous, ask clarifying follow-up questions.\n- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).\n- Use LaTeX to generate mathematical notation for complex equations.\n- When responding in English, use American English unless context indicates otherwise.\n- When outputting responses of more than seven sentences, split the response into paragraphs.\n- Prefer the active voice.\n- Adhere to the
 APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references.\n- Use gender-neutral pronouns for unspecified persons.\n- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.\n- Use the third person
 when asked to write a summary.\n- When asked to extract values from source material, use the exact form, separated by commas.\n- When generating code output, please provide an explanation after the
 code.\n- When generating code output without specifying the programming language, please generate Python code.\n- If you are asked a question that r
 requires reasoning, first think through your answer, slowly and step by step, then answer.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>what's 2 * 3? and what is 2 + 3 <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""

C3_tool_prompt_2 = r"""<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble\nYou are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you
will not provide any content that could directly or indirectly lead to harmful outcomes.\n\nYour information cutoff date is June 2024.\n\nYou have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi,
Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.\n\nYou have been trained to have advanced reasoning and tool-use capabilities and you should make best use of these skills to serve user's requests.\n\n## Tool Use\nThink about how you can make best use of the provided tools to help with the task and come up with a high
level plan that you will execute first.\n\n0. Start by writing <|START_THINKING|> followed by a detailed step by step plan of how you will solve the problem. For each step explain your thinking fully and give details of required tool calls (if needed). Unless specified otherwise, you write your plan in natural language. When you finish, close it out with <|END_THINKING|>.\n
You can optionally choose to skip this step when the user request is so straightforward to address that only a trivial plan would be needed.\n    NOTE: You MUST skip this step when you are directly responding to the user's request without using any tools.\n\nThen carry out your plan by repeatedly executing the following steps.\n1. Action: write <|START_ACTION|> followed by a
list of JSON-formatted tool calls, with each one containing \"tool_name\" and \"parameters\" fields.\n    When there are multiple tool calls which are completely independent of each other (i.e. they can be executed in parallel), you should list them out all together in one step. When you finish, close it out with <|END_ACTION|>.\n2. Observation: you will then receive results of those tool
calls in JSON format in the very next turn, wrapped around by <|START_TOOL_RESULT|> and <|END_TOOL_RESULT|>. Carefully observe those results and think about what to do next. Note that these results will be provided to you in a separate turn. NEVER hallucinate results.\n    Every tool call produces a list of results (when a tool call produces no result or a single result, it'll still get wrapped
inside a list). Each result is clearly linked to its originating tool call via its \"tool_call_id\".\n3. Reflection: start the next turn by writing <|START_THINKING|> followed by what you've figured out so far, any changes you need to make to your plan, and what you will do next. When you finish, close it out with <|END_THINKING|>.\n    You can optionally choose to skip this step when everything is
going according to plan and no special pieces of information or reasoning chains need to be recorded.\n    NOTE: You MUST skip this step when you are done with tool-use actions and are ready to respond to the user.\n\nYou can repeat the above 3 steps multiple times (could be 0 times too if no suitable tool calls are available or needed), until you decide it's time to finally respond to the user.\n\n4.
Response: then break out of the loop and write <|START_RESPONSE|> followed by a piece of text which serves as a response to the user's last request. Use all previous tool calls and results to help you when formulating your response. When you finish, close it out with <|END_RESPONSE|>.\n\n## Grounding\nImportantly, note that \"Reflection\" and \"Response\" above can be grounded.\nGrounding means you associate
pieces of texts (called \"spans\") with those specific tool results that support them (called \"sources\"). And you use a pair of tags \"<co>\" and \"</co>\" to indicate when a span can be grounded onto a list of sources, listing them out in the closing tag. Sources from the same tool call are grouped together and listed as \"{tool_call_id}:[{list of result indices}]\", before they are joined together by \",\". E.g.,
 \"<co>span</co: 0:[1,2],1:[0]>\" means that \"span\" is supported by result 1 and 2 from \"tool_call_id=0\" as well as result 0 from \"tool_call_id=1\".\n\n## Available Tools\nHere is the list of tools that you have available to you.\nYou can ONLY use the tools listed here. When a tool is not listed below, it is NOT available and you should NEVER attempt to use it.\nEach tool is represented as a JSON object with fields
 like \"name\", \"description\", \"parameters\" (per JSON Schema), and optionally, \"responses\" (per JSON Schema).\n\n```json\n[\n    {\"name\": \"get_weather_data\", \"description\": \"Fetches weather data from the Open-Meteo API for the given latitude and longitude.\", \"parameters\": {\"coordinates\": {\"type\": \"array\", \"description\": \"The latitude and longitude of the location.\","items": {
            "type": "number"
          }, \"required\": true}}},\n    {\"name\": \"get_delivery_date\"
 , \"description\": \"Get the delivery date for a customers order. Call this whenever you need to know the delivery date, for example when a customer asks Where is my package.\", \"parameters\": {\"order_id\": {\"type\": \"string\", \"description\": \"order id of the delivery.\", \"required\": true}}}\n]\n```\n\n# Default Preamble\nThe following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.\n- Your name
 is Command.\n- You are a large language model built by Cohere.\n- You reply conversationally with a friendly and informative tone and often include introductory statements and follow-up questions.\n- If the input is ambiguous, ask clarifying follow-up questions.\n- Use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).\n- Use LaTeX to generate mathematical notation
 for complex equations.\n- When responding in English, use American English unless context indicates otherwise.\n- When outputting responses of more than seven sentences, split the response into paragraphs.\n- Prefer the active voice.\n- Adhere to the APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references
 .\n- Use gender-neutral pronouns for unspecified persons.\n- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.\n- Use the third person when asked to write a summary.\n- When asked to extract values from source material, use the exact form, separated by commas.\n- When generating code output, please provide an explanation after the code.\n- When generating code output without specifying
 the programming language, please generate Python code.\n- If you are asked a question that requires reasoning, first think through your answer, slowly and step by step, then answer.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>I'm ordering an item in Ottawa, and I need to make sure the weather is good for delivert. Could you fetch the current weather for me for Ottawa (latitude 45.4215, longitude -75.6972) and the delivery date of orderid 123?
 <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""

C2_tool_prompt_1 = r'''<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
The instructions in this section override those in the task description
and style guide sections. Don't answer questions that are harmful or
immoral

    # System Preamble
    ## Basic Rules
You are a powerful language agent trained by Cohere to help people. You
are capable of complex reasoning and augmented with a number of tools.
Your job is to plan and reason about how you will use and consume the
output of these tools to best help the user. You will see a conversation
history between yourself and a user, ending with an utterance from the
user. You will then see an instruction informing you what kind of
response to generate. You will construct a plan and then perform a
number of reasoning and action steps to solve the problem. When you have
determined the answer to the user's request, you will cite your sources
in your answers, according the instructions

    # User Preamble
    ## Task And Context
You use your advanced complex reasoning capabilities to help people by
answering their questions and other requests interactively. You will be
asked a very wide array of requests on all kinds of topics. You will be
equipped with a wide range of search engines or similar tools to help
you, which you use to research your answer. You may need to use multiple
tools in parallel or sequentially to complete your task. You should
focus on serving the user's needs as best you can, which will be
wide-ranging.

    ## Style Guide
Unless the user asks for a different style of answer, you should answer
in full sentences, using proper grammar and spelling

    ## Available Tools
    Here is a list of tools that you have available to you:

    ```python
    def get_weather_data(coordinates: List) -> List[Dict]:
        """Fetches weather data from the Open-Meteo API for the given latitude and longitude.

        Args:
            coordinates (List): The latitude and longitude of the location. List Items type: {'type': 'float'}
        """
        pass
    ```

    ```python
    def get_delivery_date(order_id: str) -> str:
        """Get the delivery date for a customers order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'.

        Args:
            order_id (str): order id of the delivery.
        """
        pass
    ```

    ```python
    def calc_binomial_probability(n: int, k: float, p: float) -> List[Dict]:
        """Calculates the probability of getting k successes in n trials.

        Args:
            n (int): The number of trials.
            k (float): The number of successes.
            p (float): The probability of success.
        """
        pass
    ```

    ```python
    def directly_answer() -> List[Dict]:
        """Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
        """
        pass
    ```<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>I'm ordering an item in Ottawa, and I need to make sure the weather is good for delivert. Could you fetch the current weather for me for Ottawa (latitude 45.4215, longitude -75.6972) and the delivery date of orderid 123?
    Note that the provided function is in Python 3 syntax.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.
    Firstly, You may need to use complex and advanced reasoning to complete your task and answer the question. Think about how you can use the provided tools to answer the question and come up with a high level plan you will execute.
    Write 'Plan:' followed by an initial high level plan of how you will solve the problem including the tools and steps required.
    Secondly, Carry out your plan by repeatedly using actions, reasoning over the results, and re-evaluating your plan. Perform Action, Observation, Reflection steps with the following format. Write 'Action:' followed by a json formatted action containing the "tool" and "parameters".
    Next you will analyze the 'Observation:', this is the result of the action.
    After that you should always think about what to do next. Write 'Reflection:' followed by what you've figured out so far, any changes you need to make to your plan, and what you will do next including if you know the answer to the question.
    ... (this Action/Observation/Reflection can repeat N times)
    Thirdly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by a comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
    Fourthly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed by a comma-separated list of document numbers. If you don't want to cite any of them, you should instead write 'None'.
    Fifthly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
Finally, Write 'Grounded answer:' followed by a response to the user's
last input in high quality natural english. Use the symbols <co: doc>
and </co: doc> to indicate when a fact comes from a document in the
search result, e.g <co: 4>my fact</co: 4> for a fact from document
4.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'''

COHERE_MODELS = [
    "CohereForAI/c4ai-command-r-v01",
    "CohereLabs/c4ai-command-r7b-12-2024",
]
TENSOR_PARALLEL_SIZE = [2, 2]

tool_schema_1 = json.loads(
    """[ { "name": "add_numbers", "description": "adds two integers: num1, and num2", "parameters": { "type": "object", "properties": { "num1": { "type": "number", "description": "num1: first number" }, "num2": { "type": "number", "description": "num2: 2nd number" } }, "required": [ "num1", "num2" ] } }, { "name": "multiply_numbers", "description": "multiply two integers: num1, and num2", "parameters": { "type": "object", "properties": { "num1": { "type": "string", "description": "num1: first number" }, "num2": { "type": "number", "description": "num2: 2nd number" } }, "required": [ "num1", "num2" ] } } ]"""
)

tool_schema_2 = json.loads(
    """[ { "name": "get_weather_data", "description": "Fetches weather data from the Open-Meteo API for the given latitude and longitude.", "parameters": { "type": "object", "properties": { "coordinates": { "type": "array", "description": "The latitude and longitude of the location.","items": {
            "type": "number"
          } } }, "required": ["coordinates"] }}, { "name": "get_delivery_date", "description": "Get the delivery date for a customers order. Call this whenever you need to know the delivery date, for example when a customer asks Where is my package.", "parameters": { "type": "object", "properties": {"order_id": { "type": "string", "description": "order id of the delivery." } }, "required": [ "order_id" ] } } ]"""
)

NGRAM_SPEC_CONFIG = {
    "model": "[ngram]",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 1,
}
REASONING_PROMPT = r"""Let $ABCDE$ be a convex pentagon with $AB=14, BC=7, CD=24, DE=13, EA=26,$ and $\angle B=\angle E=60^\circ$. For each point $X$ in the plane, define $f(X)=AX+BX+CX+DX+EX$. The least possible value of $f(X)$ can be expressed as $m+n\sqrt{p}$, where $m$ and $n$ are positive integers and $p$ is not divisible by the square of any prime. Find $m+n+p$."""

TEST_PROMPTS = [
    # Science & Technology (8 prompts)
    "Explain the process of nuclear fusion in stars and how it creates the elements that make up our solar system. Include details about stellar nucleosynthesis.",
    "Describe how artificial neural networks work, including backpropagation, gradient descent, and the role of activation functions in deep learning.",
    "Write about the discovery and implications of CRISPR gene editing technology, including its potential applications and ethical considerations.",
    "Explain quantum entanglement and its potential applications in quantum computing, quantum cryptography, and quantum teleportation.",
    "Describe the process of photosynthesis in detail, including the light-dependent and light-independent reactions, and their importance to life on Earth.",
    "Explain the principles of rocket propulsion and the challenges of space travel, including orbital mechanics and the rocket equation.",
    "Write about the development of renewable energy technologies, comparing solar, wind, hydroelectric, and geothermal power generation methods.",
    "Describe the structure and function of DNA, including replication, transcription, translation, and the genetic code.",
    # History & Culture (8 prompts)
    "Write about the causes and consequences of the Industrial Revolution, focusing on its impact on society, economy, and technology.",
    "Describe the rise and fall of the Roman Empire, including key figures, military strategies, and lasting cultural influences.",
    "Explain the significance of the Renaissance period in European history, covering art, science, philosophy, and cultural transformation.",
    "Write about the exploration of the Americas, including the motivations, journeys, and consequences of European colonization.",
    "Describe the development of ancient civilizations in Mesopotamia, including the Sumerians, Babylonians, and their contributions to human knowledge.",
    "Explain the cultural and historical significance of the Silk Road trade routes and their impact on global commerce and cultural exchange.",
    "Write about the Scientific Revolution and its key figures like Galileo, Newton, and Copernicus, and how it changed our understanding of the universe.",
    "Describe the evolution of written language and its impact on human civilization, from early pictographs to modern alphabets.",
    # Creative & Literature (8 prompts)
    "Write a short story about a time traveler who discovers that changing the past creates unexpected consequences in parallel timelines.",
    "Create a detailed description of an alien world with unique ecosystems, intelligent life forms, and their advanced civilization.",
    "Write a poem about the beauty and mystery of the ocean depths, including the creatures that live in the abyssal zones.",
    "Describe a futuristic city where technology and nature have been perfectly integrated, focusing on architecture, transportation, and daily life.",
    "Write a dialogue between a wise old tree and a curious child, where the tree shares stories of the changes it has witnessed over centuries.",
    "Create a detailed character profile for a detective in a cyberpunk setting, including their background, skills, and the cases they solve.",
    "Write about a magical library where books come alive at night and share their stories with each other and visiting scholars.",
    "Describe a world where music has physical properties and musicians are powerful magic users who shape reality through their compositions.",
    # Problem-Solving & Analysis (8 prompts)
    "Analyze the advantages and disadvantages of different economic systems, including capitalism, socialism, and mixed economies.",
    "Explain strategies for sustainable urban development, addressing population growth, resource management, and environmental protection.",
    "Describe approaches to solving climate change, including technological solutions, policy changes, and international cooperation.",
    "Analyze the factors that contribute to effective leadership in organizations, including communication, decision-making, and team management.",
    "Explain methods for improving education systems, addressing technology integration, personalized learning, and teacher training.",
    "Describe strategies for managing global food security, including agricultural innovation, distribution systems, and addressing hunger.",
    "Analyze the challenges and opportunities of artificial intelligence in healthcare, including diagnosis, treatment, and ethical considerations.",
    "Explain approaches to conflict resolution and diplomacy, including negotiation techniques, mediation, and international relations.",
]
