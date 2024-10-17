""" GraphRAG researcher module"""

from textwrap import dedent

from llama_index.core.chat_engine.types import ChatMessage

from app.agents import FunctionCallingAgentConfig, WorkflowConfig
from app.agents.single import FunctionCallingAgent
from app.engine.tools.property_graph import (
    CypherQueryRetriever,
    KeywordSynonymRetriever,
    PropertyGraphTool,
    VectorSimilarityRetriever,
)

from app.config import graph_store, vector_store, llm


def _get_research_tools() -> list[PropertyGraphTool]:
    """
    Researcher takes responsibility for retrieving information
    using property graph querying strategies.
    """
    graph_store = graph_store
    vector_store = vector_store
    llm = "your_language_model_instance"

    tools = [
        KeywordSynonymRetriever(graph_store, vector_store, llm),
        VectorSimilarityRetriever(graph_store, vector_store, llm),
        CypherQueryRetriever(graph_store, vector_store, llm),
    ]
    return tools


def create_researcher(chat_history: list[ChatMessage]):
    """
    Researcher is an agent that takes responsibility for using tools to complete a given task, focusing exclusively on property graph querying strategies.
    """
    tools = _get_research_tools()
    return FunctionCallingAgent(
        name="graph_rag_researcher",
        workflow_config=WorkflowConfig(verbose=True, num_concurrent_runs=1),
        agent_config=FunctionCallingAgentConfig(name="graph_rag_researcher"),
        tools=tools,
        description="Expert in retrieving information using property graph querying strategies",
        system_prompt=dedent(
            """
            You are a researcher agent specialized in property graph querying. 
            You are given a research task.
            
            Use various graph querying strategies to retrieve information. 
            Always consider the user's request context to understand the main content needs.

            
            If you use the tools but don't find any related information, 
            please return "I didn't find any new information for {the topic}." 
            along with the content you found. Don't try to make up information yourself.
            
            If the request doesn't need any new information because it was already in the 
            conversation history, please return "The task doesn't need any new information. 
            Please reuse the existing content in the conversation history."
        """
        ),
        chat_history=chat_history,
    )
