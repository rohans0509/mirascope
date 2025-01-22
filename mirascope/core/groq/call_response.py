"""This module contains the `GroqCallResponse` class.

usage docs: learn/calls.md#handling-responses
"""

from functools import cached_property

from groq.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from groq.types.completion_usage import CompletionUsage
from pydantic import SerializeAsAny, computed_field

from ..base import BaseCallResponse, transform_tool_outputs
from ._utils import calculate_cost
from .call_params import GroqCallParams
from .dynamic_config import AsyncGroqDynamicConfig, GroqDynamicConfig
from .tool import GroqTool


class GroqCallResponse(
    BaseCallResponse[
        ChatCompletion,
        GroqTool,
        ChatCompletionToolParam,
        AsyncGroqDynamicConfig | GroqDynamicConfig,
        ChatCompletionMessageParam,
        GroqCallParams,
        ChatCompletionUserMessageParam,
    ]
):
    """A convenience wrapper around the Groq `ChatCompletion` response.

    When calling the Groq API using a function decorated with `groq_call`, the
    response will be an `GroqCallResponse` instance with properties that allow for
    more convenience access to commonly used attributes.

    Example:

    ```python
    from mirascope.core import prompt_template
    from mirascope.core.groq import groq_call


    @groq_call("llama-3.1-8b-instant")
    def recommend_book(genre: str) -> str:
        return f"Recommend a {genre} book"

    response = recommend_book("fantasy")  # response is an `GroqCallResponse` instance
    print(response.content)
    ```
    """

    _provider = "groq"

    @property
    def content(self) -> str:
        """Returns the content of the chat completion for the 0th choice."""
        message = self.response.choices[0].message
        return message.content if message.content is not None else ""

    @property
    def finish_reasons(self) -> list[str]:
        """Returns the finish reasons of the response."""
        return [str(choice.finish_reason) for choice in self.response.choices]

    @property
    def model(self) -> str:
        """Returns the name of the response model."""
        return self.response.model

    @property
    def id(self) -> str:
        """Returns the id of the response."""
        return self.response.id

    @property
    def usage(self) -> CompletionUsage | None:
        """Returns the usage of the chat completion."""
        return self.response.usage

    @property
    def input_tokens(self) -> int | None:
        """Returns the number of input tokens."""
        return self.usage.prompt_tokens if self.usage else None

    @property
    def output_tokens(self) -> int | None:
        """Returns the number of output tokens."""
        return self.usage.completion_tokens if self.usage else None

    @property
    def cost(self) -> float | None:
        """Returns the cost of the call."""
        return calculate_cost(self.input_tokens, self.output_tokens, self.model)

    @computed_field
    @cached_property
    def message_param(self) -> SerializeAsAny[ChatCompletionAssistantMessageParam]:
        """Returns the assistants's response as a message parameter."""
        message_param = self.response.choices[0].message.model_dump(
            exclude={"function_call"}
        )
        return ChatCompletionAssistantMessageParam(**message_param)

    @computed_field
    @cached_property
    def tools(self) -> list[GroqTool] | None:
        """Returns any available tool calls as their `GroqTool` definition.

        Raises:
            ValidationError: if a tool call doesn't match the tool's schema.
        """
        tool_calls = self.response.choices[0].message.tool_calls
        if not self.tool_types or not tool_calls:
            return None

        extracted_tools = []
        for tool_call in tool_calls:
            for tool_type in self.tool_types:
                if tool_call.function.name == tool_type._name():
                    extracted_tools.append(tool_type.from_tool_call(tool_call))
                    break

        return extracted_tools

    @computed_field
    @cached_property
    def tool(self) -> GroqTool | None:
        """Returns the 0th tool for the 0th choice message.

        Raises:
            ValidationError: if the tool call doesn't match the tool's schema.
        """
        if tools := self.tools:
            return tools[0]
        return None

    @classmethod
    @transform_tool_outputs
    def tool_message_params(
        cls, tools_and_outputs: list[tuple[GroqTool, str]]
    ) -> list[ChatCompletionToolMessageParam]:
        """Returns the tool message parameters for tool call results.

        Args:
            tools_and_outputs: The list of tools and their outputs from which the tool
                message parameters should be constructed.

        Returns:
            The list of constructed `ChatCompletionToolMessageParam` parameters.
        """
        return [
            ChatCompletionToolMessageParam(
                role="tool",
                content=output,
                tool_call_id=tool.tool_call.id,
                name=tool._name(),  # pyright: ignore [reportCallIssue]
            )
            for tool, output in tools_and_outputs
        ]

def calculate_cost(
    input_tokens: int | None, output_tokens: int | None, model: str
) -> float | None:
    """Calculate the cost of the API call based on tokens used and model.
    
    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
        model: Name of the model used
    
    Returns:
        Cost in USD, or None if tokens are None
    """
    if input_tokens is None or output_tokens is None:
        return None

    # Pricing per million tokens (input, output)
    MODEL_COSTS = {
        "llama-3.2-1b-preview": (0.04, 0.04),
        "llama-3.2-3b-preview": (0.06, 0.06),
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "llama3-70b-8192": (0.59, 0.79),
        "llama3-8b-8192": (0.05, 0.08),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "gemma2-9b-it": (0.20, 0.20),
        "llama-guard-3-8b": (0.20, 0.20),
        "llama-3.3-70b-specdec": (0.59, 0.99),
    }

    if model not in MODEL_COSTS:
        return None

    input_cost, output_cost = MODEL_COSTS[model]
    total_cost = (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000
    return total_cost
