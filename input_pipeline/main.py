#!/usr/bin/env python3
"""
Build knowledge graph from Markdown files
"""
import pickle
import time
from typing import Any, Dict, List, Tuple, TypedDict
import logging


import yaml  # type: ignore [import-untyped]
from llama_index.core import (  # type: ignore [import-untyped]
    PropertyGraphIndex,
    Settings,
    SimpleDirectoryReader,
    schema,
)
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.azure_openai import (  # type: ignore [import-untyped]
    AzureOpenAIEmbedding,
)
from llama_index.graph_stores.neo4j import (  # type: ignore [import-untyped]
    Neo4jPropertyGraphStore,
)
from llama_index.llms.azure_openai import AzureOpenAI  # type: ignore [import-untyped]
from llama_parse import LlamaParse, ResultType  # type: ignore [import-untyped]

from module_settings import settings, logger

Triple = Tuple[str, str, str]


class EntitiesConfig(TypedDict):
    entities: List[str]
    relations: List[str]
    validation_schema: Dict[str, List[str]]


graph_store = Neo4jPropertyGraphStore(
    username=settings.neo4j_username,
    password=settings.neo4j_password.get_secret_value(),
    url=settings.neo4j_uri,
)

logger.debug("Setting up parser")
# set up parser
parser = LlamaParse(
    # result_type=ResultType.MD,
    split_by_page=True,  # force to split by page
    api_key=settings.llama_cloud_api_key.get_secret_value(),
)

logger.debug("Setting up LLM")
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

logger.debug("Setting up embedding model")
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
    logger.debug("Building knowledge graph ...")
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
    logger.info("Loaded %d entities", len(entities))  # type: ignore [annotation-unchecked]

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
            logger.debug("Validating %s -> %s", entity, ", ".join(relations))

    logger.debug("Setting up knowledge graph extractor")
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

    # Use Llama-Parser to extract text from PDF files
    file_extractor = {".pdf": parser}
    # only load pdf files
    required_exts = [".pdf"]
    documents: list[schema.Document] = []  # type: ignore [annotation-unchecked]
    documents = SimpleDirectoryReader(
        settings.pdf_path,
        file_extractor=file_extractor,
        required_exts=required_exts,
        recursive=False,
    ).load_data()
    logger.debug("Loaded %d documents", len(documents))

    # Save documents for further inspection
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        with open("documents.pkl", "wb") as file:
            # Use pickle.dump to serialize and save the documents
            logger.debug("Saving documents to disk")
            pickle.dump(documents, file, protocol=pickle.HIGHEST_PROTOCOL)

    build_knowledge_graph(documents, kg_extractor, show_progress=False)

    logger.info("Done")


if __name__ == "__main__":
    start_time = time.time()
    main()
    mins, secs = divmod(time.time() - start_time, 60)
    hrs, mins = divmod(mins, 60)
    logger.info("Execution time: %02d:%02d:%02d", hrs, mins, secs)
