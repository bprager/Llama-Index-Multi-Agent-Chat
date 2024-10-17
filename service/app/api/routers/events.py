import logging

from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging_level),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s:%(lineno)d: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("events.log", mode="w"),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)
