from textwrap import dedent
from typing import List, Optional

from llama_index.core.chat_engine.types import ChatMessage

from app.agents import AgentCallingAgent


def create_choreography(
    chat_history: Optional[List[ChatMessage]] = None,
) -> AgentCallingAgent:
    return "Choreography created"
