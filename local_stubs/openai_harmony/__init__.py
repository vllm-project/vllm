class Role:
    ASSISTANT = 'assistant'

class Author:
    def __init__(self, role=None):
        self.role = role

class TextContent:
    def __init__(self, text=''):
        self.text = text

class Message:
    def __init__(self, author=None, content=None, channel=None, recipient=None, content_type=None):
        self.author = author
        self.content = content or []
        self.channel = channel
        self.recipient = recipient
        self.content_type = content_type

class ToolDescription: ...
class ToolNamespaceConfig: ...
