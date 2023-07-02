# From https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""
Conversation prompt templates.
"""

import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    CHATGLM = auto()
    CHATML = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if self.system == "" else self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if self.system:
                ret = self.system + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna template
register_conv_template(
    Conversation(
        name="vicuna",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# ChatGLM default template
register_conv_template(
    Conversation(
        name="chatglm",
        system="",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n",
    )
)

# ChatGLM2 default template
register_conv_template(
    Conversation(
        name="chatglm2",
        system="",
        roles=("问", "答"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n\n",
    )
)

# Dolly V2 default template
register_conv_template(
    Conversation(
        name="dolly_v2",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        system="",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

# OpenAssistant default template
register_conv_template(
    Conversation(
        name="oasst_llama",
        system="",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Tulu default template
register_conv_template(
    Conversation(
        name="tulu",
        system="",
        roles=("<|user|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        name="stablelm",
        system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        name="baize",
        system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        name="rwkv",
        system="",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        name="openbuddy",
        system="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        name="phoenix",
        system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt-7b-chat",
        system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# MPT-30b-chat default template
register_conv_template(
    Conversation(
        name="mpt-30b-chat",
        system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# MPT-30b-instruct default template
# reference: https://huggingface.co/mosaicml/mpt-30b-instruct#formatting
register_conv_template(
    Conversation(
        name="mpt-30b-instruct",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        stop_token_ids=[50278, 0],
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40
register_conv_template(
    Conversation(
        name="bard",
        system="",
        roles=("0", "1"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        name="billa",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        name="redpajama-incite",
        system="",
        roles=("<human>", "<bot>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        system="",
        roles=("<|prompt|>", "<|answer|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Robin default template
register_conv_template(
    Conversation(
        name="Robin",
        system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("###Human", "###Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n",
        stop_token_ids=[2, 396],
        stop_str="###",
    )
)

# Snoozy default template
# Reference: https://github.com/nomic-ai/gpt4all/blob/d4861030b778da6db59d21d2927a4aba4f9f1f43/gpt4all-bindings/python/gpt4all/gpt4all.py#L232
register_conv_template(
    Conversation(
        name="snoozy",
        system="### Instruction:\nThe prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.",
        roles=("### Prompt", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)

# manticore default template
register_conv_template(
    Conversation(
        name="manticore",
        system="",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# Falcon default template
register_conv_template(
    Conversation(
        name="falcon",
        system="",
        roles=("User", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.RWKV,
        sep="\n",
        sep2="<|endoftext|>",
        stop_str="\nUser",  # use stop_str to stop generation after stop_token_ids, it will also remove stop_str from the generated text
        stop_token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],  # it better only put special tokens here, because tokenizer only remove special tokens
    )
)

# ChagGPT default template
register_conv_template(
    Conversation(
        name="polyglot_changgpt",
        system="",
        roles=("B", "A"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# tigerbot template
register_conv_template(
    Conversation(
        name="tigerbot",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
        stop_str="###",
    )
)


if __name__ == "__main__":
    conv = get_conv_template("vicuna")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
