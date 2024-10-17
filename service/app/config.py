""" Configuration settings for the web service """

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.graph_stores.neo4j import Neo4jPGStore  # type: ignore [import-untyped]
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
    agent_type: str = "ORCHESTRATOR"
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


settings = ModelSettings()  # type: ignore [call-arg]

# graph store instance
graph_store = Neo4jPGStore(
    username=settings.neo4j_username,
    password=settings.neo4j_password.get_secret_value(),
    url=settings.neo4j_uri,
    embedding_dimension=1536,
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
    temperature=0.0,
    azure_endpoint="https://.openai.azure.com/",
    api_key="",
    api_version="2023-07-01-preview",
)
