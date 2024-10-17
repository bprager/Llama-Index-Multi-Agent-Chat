import logging

from fastapi import APIRouter, BackgroundTasks, Request, status, HTTPException
from api.routers.events import EventCallbackHandler
from app.api.routers.models import ( 
    ChatData,
)
from app.config import settings


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

cat_router = r = APIRouter()

@r.post("")
async def chat(
    request: Request,
    data: ChatData,
    background_tasks: BackgroundTasks,
):
    try:
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages(include_agent_messages=True)
        
        event_handler = EventCallbackHandler()
        engine = get_chat_engine(chat_history=messages)
        
