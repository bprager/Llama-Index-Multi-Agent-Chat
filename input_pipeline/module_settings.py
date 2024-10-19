""" module settings """

import logging
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Settings
class ModelSettings(BaseSettings):
    """Settings for knowledge graph builder"""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )
    api_type: str
    azure_openai_api_key: SecretStr
    azure_openai_api_version: str
    azure_openai_embedding_deployment: str
    azure_openai_embedding_model: str
    azure_openai_endpoint: str
    azure_openai_model: str = "gpt-4-turbo"
    llama_cloud_api_key: SecretStr
    logging_level: str = "INFO"
    markdown_path: str
    neo4j_password: SecretStr
    neo4j_uri: str
    neo4j_username: str
    pdf_path: str


settings = ModelSettings()  # type: ignore [call-arg]

# Obtain a named logger
logger = logging.getLogger("input_pipeline")
logger.setLevel(logging.DEBUG)  # Set the logging level

# Define the formatter with the specified format and date format
formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create the handlers as specified
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler("input_pipeline.log", mode="w")

# Set the logging level for handlers (optional, as they inherit from logger)
stream_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Assign the formatter to the handlers
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to your logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# Loggers exlude from debug
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("fsspec").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
logging.getLogger("neo4j").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO)
