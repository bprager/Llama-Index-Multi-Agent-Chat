#!/usr/bin/env python3
"""
Build knowledge graph from Markdown files
"""
import logging
import pickle
import time
from typing import Any, Dict, List, Tuple, TypedDict

import yaml  # type: ignore [import-untyped]
from llama_index.core import PropertyGraphIndex, Settings  # type: ignore [import-untyped]
from llama_index.core import SimpleDirectoryReader, schema  # type: ignore [import-untyped]
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.azure_openai import AzureOpenAI
from llama_parse import LlamaParse, ResultType  # type: ignore [import-untyped]
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Triple = Tuple[str, str, str]


class EntitiesConfig(TypedDict):
    entities: List[str]
    relations: List[str]
    validation_schema: Dict[str, List[str]]


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
    azure_openai_path: str
    llama_cloud_api_key: SecretStr
    logging_level: str = "INFO"
    markdown_path: str
    neo4j_password: SecretStr
    neo4j_uri: str
    neo4j_username: str
    pdf_path: str


settings = ModelSettings()  # type: ignore [call-arg]

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging_level),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s:%(lineno)d: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("input_pipeline.log", mode="w"),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Loggers exlude from debug
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("fsspec").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.INFO)
logging.getLogger("neo4j").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO)

graph_store = Neo4jPropertyGraphStore(
    username=settings.neo4j_username,
    password=settings.neo4j_password.get_secret_value(),
    url=settings.neo4j_uri,
)

# set up parser
parser = LlamaParse(
    result_type=ResultType.MD,
    split_by_page=True,  # force to split by page
    api_key=settings.llama_cloud_api_key.get_secret_value(),
)

# set up LLM
llm = AzureOpenAI(
    api_key=settings.azure_openai_api_key.get_secret_value(),
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    model=settings.azure_openai_model,
    deployment_name=settings.azure_openai_model,
    timeout=300,
    temperature=0.0,
)

embed_model = AzureOpenAIEmbedding(
    api_key=settings.azure_openai_api_key.get_secret_value(),
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    model=settings.azure_openai_embedding_model,
    deployment_name=settings.azure_openai_embedding_deployment,
)

Settings.llm = llm
Settings.embed_model = embed_model


def build_knowledge_graph(
    documents: list[schema.Document],
    kg_extractor: SchemaLLMPathExtractor,
    show_progress: bool = True,
) -> None:
    """Build knowledge graph from documents and store in graph store"""
    logging.debug("Building knowledge graph ...")
    PropertyGraphIndex.from_documents(
        llm=llm,
        documents=documents,
        kg_extractor=[kg_extractor],
        embed_model=embed_model,
        property_graph_store=graph_store,
        show_progress=show_progress,
    )


def load_entities(file_path: str) -> EntitiesConfig:
    """Load suggested entities for the knowledge graph from a file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    """Main function"""

    entities_config: EntitiesConfig  # type: ignore [annotation-unchecked]
    entities_config = load_entities("entity_relations.yaml")

    # Directly assign the lists from the configuration
    entities: List[str] = entities_config["entities"]  # type: ignore [annotation-unchecked]
    relations: List[str] = entities_config["relations"]  # type: ignore [annotation-unchecked]
    logging.info("Loaded %d entities", len(entities))  # type: ignore [annotation-unchecked]

    validation_schema: Dict[str, List[str]]  # type: ignore [annotation-unchecked]
    validation_schema = entities_config["validation_schema"]  # type: ignore [annotation-unchecked]

    # Convert validation_schema to a List[Triple]
    triples: List[Triple] = []  # type: ignore [annotation-unchecked]
    for entity, relations_list in validation_schema.items():
        for relation in relations_list:
            # Assuming the target entities are all entities
            for target_entity in entities:
                triples.append((entity, relation, target_entity))

    # Use Any to bypass type checking or use type: ignore
    possible_entities: Any = entities  # type: ignore [annotation-unchecked]
    possible_relations: Any = relations  # type: ignore [annotation-unchecked]

    # Use the schema to validate the relationships
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        for entity, relations in validation_schema.items():
            # Log the entity and its associated relations
            logging.debug("Validating %s -> %s", entity, ", ".join(relations))

    # Knowledge graph
    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=possible_entities,
        possible_relations=possible_relations,
        kg_validation_schema=triples,
        # if false, allows for values outside of the schema
        # useful for using the schema as a suggestion
        strict=False,
    )

    # only load pdf files
    required_exts = [".pdf"]
    documents: list[schema.Document] = []  # type: ignore [annotation-unchecked]
    documents = SimpleDirectoryReader(
        settings.pdf_path,
        required_exts=required_exts,
        recursive=False,
    ).load_data()
    logging.debug("Loaded %d documents", len(documents))

    # Save documents for further inspection
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        with open("documents.pkl", "wb") as file:
            # Use pickle.dump to serialize and save the documents
            logging.debug("Saving documents to disk")
            pickle.dump(documents, file, protocol=pickle.HIGHEST_PROTOCOL)

    build_knowledge_graph(documents, kg_extractor, show_progress=False)

    logging.info("Done")


if __name__ == "__main__":
    start_time = time.time()
    main()
    mins, secs = divmod(time.time() - start_time, 60)
    hrs, mins = divmod(mins, 60)
    logging.info("Execution time: %02d:%02d:%02d", hrs, mins, secs)
