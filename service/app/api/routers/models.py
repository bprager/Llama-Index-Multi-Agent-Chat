from typing import Any, Dict, List, Literal, Optional

from lama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field, fieldvalidator


class AgentAnnotation(BaseModel):
    agent: str
    text: str


class ArtifactAnnotation(BaseModel):
    toolCall: Dict[str, Any]
    toolOutput: Dict[str, Any]


class Annotation(BaseModel):
    type: str
    data: List[str] | AgentAnnotation | ArtifactAnnotation


class Message(BaseModel):
    role: MessageRole
    content: str
    annotations: List[Annotation] | None = None


class ChatData(BaseModel):
    messages: List[Message]
    data: Any = None

    @fieldvalidator("messages")
    def message_must_not_be_empty(self, cls, value):
        if not value:
            raise ValueError("Messages must not be empty")
        return value

    def get_last_message_content(self) -> str:
        if len(self.messages) == 0:
            raise ValueError("No messages in chat data")
        last_message = self.messages[-1]
        message_content = last_message.content.strip()
