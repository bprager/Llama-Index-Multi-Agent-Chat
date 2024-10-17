from abc import abstractmethod
from typing import Any, AsyncGenerator, List, Optional, cast

from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings as LlamaIndexSettings
from llama_index.core.tools import FunctionTool, ToolOutput, ToolSelection
from llama_index.core.tools.types import AsyncBaseTool, BaseTool
from llama_index.core.workflow import (Context, Event, StartEvent, StopEvent,
                                       Workflow, step)
from llama_index.core.workflow.service import ServiceManager
from pydantic import BaseModel


class InputEvent(Event):
    input: list[ChatMessage]


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class AgentRunEvent(Event):
    name: str
    _msg: str

    @property
    def msg(self):
        return self._msg

    @msg.setter
    def msg(self, value):
        self._msg = value


class AgentRunResult(BaseModel):
    response: ChatResponse
    sources: list[ToolOutput]


class ContextAwareTool(FunctionTool):
    @abstractmethod
    async def accall(self, ctx: Context, input: Any) -> ToolOutput:
        pass


class WorkflowConfig:
    def __init__(
        self,
        verbose: bool = False,
        timeout: Optional[float] = 10.0,
        disable_validation: bool = False,
        num_concurrent_runs: Optional[int] = None,
    ):
        self.verbose = verbose
        self.timeout = timeout
        self.disable_validation = disable_validation
        self.num_concurrent_runs = num_concurrent_runs


class FunctionCallingAgentConfig:
    def __init__(
        self, name: str, write_events: bool = True, description: Optional[str] = None
    ):
        self.name = name
        self.write_events = write_events
        self.description = description


class FunctionCallingAgent(Workflow):
    def __init__(
        self,
        *_args,
        llm: Optional["FunctionCallingLLM"] = None,
        chat_history: Optional[List["ChatMessage"]] = None,
        tools: Optional[List["BaseTool"]] = None,
        system_prompt: Optional[str] = None,
        workflow_config: WorkflowConfig,
        agent_config: FunctionCallingAgentConfig,
        service_manager: Optional[ServiceManager] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            timeout=workflow_config.timeout,
            disable_validation=workflow_config.disable_validation,
            verbose=workflow_config.verbose,
            service_manager=service_manager,
            num_concurrent_runs=workflow_config.num_concurrent_runs,
        )
        self.chat_history = chat_history
        self.tools = tools or []
        self.name = agent_config.name
        self.write_events = agent_config.write_events
        self.description = agent_config.description
        self.other_args = kwargs

        if llm is None:
            llm = cast(FunctionCallingLLM, LlamaIndexSettings.llm)
        self.llm = llm
        assert self.llm.metadata.is_function_calling_model

        self.system_prompt = system_prompt

        self.memory = ChatMemoryBuffer.from_defaults(
            llm=self.llm,
            chat_history=chat_history,
            verbose=workflow_config.verbose,
            timeout=workflow_config.timeout,
        )
        self.sources: list = []

    @step
    async def prepare_chat_history(self, ctx: Context, event: StartEvent) -> InputEvent:
        # clear sources
        self.sources = []
        # set system prompt
        if self.system_prompt is not None:
            system_msg = ChatMessage(
                content=self.system_prompt, role=MessageRole.SYSTEM
            )
            self.memory.put(system_msg)
        # set streaming
        ctx.data["streaming"] = getattr(event, "streaming", False)
        # get user input
        user_input = event.input
        user_msg = ChatMessage(content=user_input, role=MessageRole.USER)
        self.memory.put(user_msg)
        if self.write_events:
            ctx.write_event_to_stream(
                AgentRunEvent(name=self.name, _msg=f"Start to work on: {user_input}")
            )
        # get chat history
        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step()
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        if ctx.data["streaming"]:
            return await self.handle_llm_input_stream(ctx, ev)

        chat_history = ev.input

        response = await self.llm.achat_with_tools(
            self.tools, chat_history=chat_history
        )
        self.memory.put(response.message)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            if self.write_events:
                ctx.write_event_to_stream(
                    AgentRunEvent(name=self.name, _msg="Finished task")
                )
            return StopEvent(
                result=AgentRunResult(response=response, sources=[*self.sources])
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    async def handle_llm_input_stream(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        async def response_generator() -> AsyncGenerator:
            response_stream = await self.llm.astream_chat_with_tools(
                self.tools, chat_history=chat_history
            )

            full_response = None
            yielded_indicator = False
            async for chunk in response_stream:
                if "tool_calls" not in chunk.message.additional_kwargs:
                    # Yield a boolean to indicate whether the response is a tool call
                    if not yielded_indicator:
                        yield False
                        yielded_indicator = True

                    # if not a tool call, yield the chunks!
                    yield chunk
                elif not yielded_indicator:
                    # Yield the indicator for a tool call
                    yield True
                    yielded_indicator = True

                full_response = chunk

            # Write the full response to memory
            if full_response is not None:
                self.memory.put(full_response.message)

            # Yield the final response
            yield full_response

        # Start the generator
        generator = response_generator()

        # Check for immediate tool call
        is_tool_call = await generator.__anext__()
        if is_tool_call:
            full_response = await generator.__anext__()
            tool_calls = self.llm.get_tool_calls_from_response(full_response)
            return ToolCallEvent(tool_calls=tool_calls)

        # If we've reached here, it's not an immediate tool call, so we return the generator
        if self.write_events:
            ctx.write_event_to_stream(
                AgentRunEvent(name=self.name, _msg="Finished task")
            )
        return StopEvent(result=generator)

    @step()
    async def handle_tool_calls(self, ctx: Context, ev: ToolCallEvent) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            name = tool.metadata.get_name() if tool and tool.metadata else ""
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": name,
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                if isinstance(tool, ContextAwareTool):
                    # inject context for calling an context aware tool
                    tool_output = await cast(ContextAwareTool, tool).accall(
                        ctx=ctx, **tool_call.tool_kwargs
                    )
                else:
                    tool_output = await cast(AsyncBaseTool, tool).acall(
                        **tool_call.tool_kwargs
                    )
                self.sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        for msg in tool_msgs:
            self.memory.put(msg)

        chat_history = self.memory.get()
        return InputEvent(input=chat_history)
