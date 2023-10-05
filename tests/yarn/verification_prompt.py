TEMPLATE = '''\
This is a conversation between a User and an Assistant:
User: {system}
Assistant: Got it. How can I help you?
User: {instruction}
Assistant: \
'''

SYSTEM = '''\
The Assistant follows the instructions precisely and always provides
an accurate and concise, but still complete answer every time. \
'''

INSTRUCTION = '''\
What are the first 10 steps to troubleshoot a PC which cannot boot into Windows?
'''

PROMPT = TEMPLATE.format(system=SYSTEM, instruction=INSTRUCTION)
