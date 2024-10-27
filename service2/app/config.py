""" Configuration settings for the web service """

import logging
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPGStore  # type: ignore [import-untyped]
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.llms.azure_openai import AzureOpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Settings
class ModelSettings(BaseSettings):
    """Settings for knowledge graph builder"""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )
    agent_type: str = "ORCHESTRATOR"  # or "CHOREOGRAPHY"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    azure_openai_api_key: SecretStr
    azure_openai_api_version: str
    azure_openai_embedding_deployment: str
    azure_openai_embedding_model: str
    azure_openai_endpoint: str
    azure_openai_engine: str
    azure_openai_llm_deployment: str
    azure_openai_model: str
    embedding_dimension: int
    conversation_starters: str
    environment: str = "dev"
    llama_cloud_api_key: SecretStr
    llm_temperature: float
    logging_level: str = "INFO"
    neo4j_password: SecretStr
    neo4j_uri: str
    neo4j_username: str
    top_k: int
    verbose: bool = False


settings = ModelSettings()  # type: ignore [call-arg]

# graph store instance
graph_store = Neo4jPGStore(
    username=settings.neo4j_username,
    password=settings.neo4j_password.get_secret_value(),
    url=settings.neo4j_uri,
)

# vector store instance
vector_store = Neo4jVectorStore(
    username=settings.neo4j_username,
    password=settings.neo4j_password.get_secret_value(),
    url=settings.neo4j_uri,
    embedding_dimension=1536,
)

# language model instance
llm = AzureOpenAI(
    engine=settings.azure_openai_engine,
    model=settings.azure_openai_llm_deployment,
    temperature=settings.llm_temperature,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key.get_secret_value(),
    api_version=settings.azure_openai_api_version,
)

# embedding model
embed_model = AzureOpenAIEmbedding(
    model=settings.azure_openai_embedding_model,
    deployment_name=settings.azure_openai_embedding_deployment,
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key.get_secret_value(),
    api_version=settings.azure_openai_api_version,
)

Settings.llm = llm
Settings.embed_model = embed_model

# Obtain a named logger
logger = logging.getLogger("chast_service")
logger.setLevel(logging.DEBUG)  # Set the logging level

# Define the formatter with the specified format and date format
formatter = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create the handlers as specified
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler("agentic_service.log", mode="w")

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
# logging.getLogger("asyncio").setLevel(logging.INFO)
# logging.getLogger("fsspec").setLevel(logging.INFO)
# logging.getLogger("httpcore").setLevel(logging.INFO)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("llama_index").setLevel(logging.INFO)
# logging.getLogger("neo4j").setLevel(logging.INFO)
# logging.getLogger("openai").setLevel(logging.INFO)
