# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.entrypoints.openai.parser.sentence import Author, Role, Sentence, TextContent


class StreamableParser:
    """Incremental parser over completion tokens."""

    def __init__(self, *, tokenizer):
        self.sentences: list[Sentence] = []
        self.tokens: list[int] = []
        self.tokenizer = tokenizer

        # start like this
        self.current_role = Role.ASSISTANT
        self.current_sentence = Sentence(
            author=Author(role=self.current_role), content=[]
        )
        self.current_channel = "think"
        self.current_text = ""

    def render_for_completion():
        """Maybe this can be the chat template to help generate the initial prompt?"""

        pass

    def process(self, token: int) -> "StreamableParser":
        # see process_next()
        # https://github.com/openai/harmony/blob/main/src/encoding.rs#L1114
        self.tokens.append(token)
        decoded = self.tokenizer.decode(token)
        if token == 200051:  # </think>
            new_content = TextContent(
                text=self.current_text, channel=self.current_channel
            )
            self.current_sentence.content.append(new_content)

            self.current_text = ""
            self.current_channel = "final"
        elif token == 200020:
            # end of sentence
            new_content = TextContent(
                text=self.current_text, channel=self.current_channel
            )
            self.current_sentence.content.append(new_content)
            self.sentences.append(self.current_sentence)

            self.current_text = ""
            self.current_channel = None
        else:
            self.current_text += decoded

        # TODO: current state of sentences, etc
        return self


def get_streamable_parser_for_simple_context(*, tokenizer) -> StreamableParser:
    return StreamableParser(tokenizer=tokenizer)


"""
TODO:
how to figure out which tokens are special tokens

system
tool
ai
"""
