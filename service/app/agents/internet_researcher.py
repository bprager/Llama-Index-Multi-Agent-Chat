""" Internet researcher agent module. """

from textwrap import dedent
from typing import List, Optional

from llama_index.core.chat_engine.types import ChatMessage
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from app.agents import FunctionCallingAgentConfig, WorkflowConfig
from app.agents.single import FunctionCallingAgent
from app.config import settings
from app.engine.index import get_index
from app.engine.tools import ToolFactory


def _create_query_engine_tool() -> QueryEngineTool | None:
    """Provide an agent worker that can be used to query the index."""
    index = get_index()
    if index is None:
        return None
    top_k = settings.top_k
    query_engine = index.as_query_engine(
        similarity_top_k=top_k, chat_mode=True, verbose=settings.verbose
    )
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_index",
            description=dedent(
                """
                Use this tool to retrieve information 
                about the documents from the index.
                """
            ).strip(),
        ),
    )


def _get_research_tools() -> QueryEngineTool:
    """
    The Internet researcher takes responsibility for
    retrieving information from various piblic Internet sites.
    Try all available tools available to find the answer.
    """
    tools = []
    query_engine_tool = _create_query_engine_tool()
    if query_engine_tool is not None:
        tools.append(query_engine_tool)
    researcher_tool_names = ["duckduckgo", "wikipedia.WikipediaToolSpec"]
    configured_tools = ToolFactory.from_env(use_map=True)
    for tool in configured_tools.items():
        if tool.name in researcher_tool_names:
            tools.extend(tool)
    return tools


def create_internet_researcher(
    chat_history: Optional[List[ChatMessage]] = None,
) -> FunctionCallingAgent:
    """
    The Internet researcher is an agent that takes responsibility for using tools to complete a given task.
    """
    tools = _get_research_tools()
    return FunctionCallingAgent(
        name="internet_researcher",
        tools=tools,
        workflow_config=WorkflowConfig(verbose=settings.verbose, num_concurrent_runs=1),
        agent_config=FunctionCallingAgentConfig(name="internet_researcher"),
        description=dedent(
            """
            Researcher in retrieving any unknown content or searching for information from the Internet.
            """
        ).strip(),
        system_prompt=dedent(
            """
            You are an expert in retrieving information. 
            You are given a information aqcuisition task.
            
            If the conversation already includes the information 
            and there is no need to search for it, you should 
            provide the answer directly.  Otherwise, you must use 
            tools to retieve information needed to complete the task.
            
            It's normal for the task to include some ambiguity or missing information.
            You must always think carefully about the context and the task at hand to
            understand what is the main content to be retrieved.
            
            If you use the tools but don't find any related information, you should
            return "I din't find any information for the {topic}" as the final answer
            along with the content you found. Don't make up any information yourself.
            If the request doesn't need any new information, because it was in the
            conversation history, you should return "The tasl doesn't need any new information.
            Please reuse the existing content in the conversation history."
            """
        ).strip(),
        chat_history=chat_history,
    )
