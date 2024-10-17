import logging
from typing import List, Optional
from app.config import settings
from app.agents import create_choreography, create_workflow
from llama_index.core.chat_engine.types import ChatMessage
from llama_index.core.llms import Workflow

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging_level),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s:%(lineno)d: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chat.log", mode="w"),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_chat_engine(
    chat_history: Optional[List[ChatMessage]] = None,
    **kwargs,
) -> Workflow:
    match settings.agent_type:
        case "CHOREOGRAPHY":  # agents decide themselves what to do
            agent = create_choreography(chat_history=chat_history)
        case _:
            agent = create_workflow(chat_history=chat_history)

    logging.info("Using agent: %s", agent.name)
