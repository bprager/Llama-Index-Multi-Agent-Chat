from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.core.schema import (
    NodeWithScore,
    QueryBundle,
)

from llama_index.core.indices.property_graph import CypherTemplateRetriever
from llama_index.core.bridge.pydantic import BaseModel, Field

from textwrap import dedent
from app.config import settings


class PropertyGraphTool:
    def __init__(self, graph_store, vector_store, llm):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm = llm

    def retrieve(self, query) -> list[NodeWithScore]:
        raise NotImplementedError("Subclasses must implement the retrieve method.")


class KeywordSynonymRetriever(PropertyGraphTool):
    def retrieve(self, query) -> list[NodeWithScore]:
        """Use this function to get content from the graph by keyword synonyms."""
        sub_retriever = LLMSynonymRetriever(self.graph_store, llm=self.llm)
        query_bundle = QueryBundle(query_str=query)
        return sub_retriever.retrieve_from_graph(query_bundle)


class VectorSimilarityRetriever(PropertyGraphTool):
    def retrieve(self, query) -> list[NodeWithScore]:
        """Use this function to get content from the graph by vector similarity."""
        sub_retriever = VectorContextRetriever(
            self.graph_store,
            vector_store=self.vector_store,
            embed_model=settings.EMBEDDING_MODEL,
        )
        query_bundle = QueryBundle(query_str=query)
        return sub_retriever.retrieve_from_graph(query_bundle)


class CypherQueryRetriever(PropertyGraphTool):
    def retrieve(self, query) -> list[NodeWithScore]:
        class Params(BaseModel):
            """Parameters for a cypher query."""

            names: list[str] = Field(
                description="A list of possible entity names or keywords related to the query."
            )

        cypher_query = dedent(
            """
        MATCH (c:Chunk)-[:MENTIONS]->(o) 
        WHERE o.name IN $names
        RETURN c.text, o.name, o.label;
        """
        )

        sub_retriever = CypherTemplateRetriever(
            self.graph_store,
            Params,
            cypher_query,
            llm=settings.llm,
        )
        query_bundle = QueryBundle(query_str=query)
        return sub_retriever.retrieve_from_graph(query_bundle)
