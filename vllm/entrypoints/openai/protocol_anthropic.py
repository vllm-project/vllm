from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AnthropicMessageBlock(BaseModel):
    role: str  # "user" | "assistant"
    content: Any

class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: List[AnthropicMessageBlock]
    max_tokens: int
    system: Optional[str] = None
    # Add further optional fields per API docs

class AnthropicMessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Dict[str, Any]]
    model: str
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    usage: Dict[str, int]
