"""The `groq_call` decorator for functions as LLM calls."""

from ..base import call_factory
from ._utils import (
    get_json_output,
    handle_stream,
    handle_stream_async,
    setup_call,
)
from .call_params import GroqCallParams
from .call_response import GroqCallResponse
from .call_response_chunk import GroqCallResponseChunk
from .dynamic_config import GroqDynamicConfig
from .stream import GroqStream
from .tool import GroqTool

groq_call = call_factory(
    TCallResponse=GroqCallResponse,
    TCallResponseChunk=GroqCallResponseChunk,
    TDynamicConfig=GroqDynamicConfig,
    TToolType=GroqTool,
    TStream=GroqStream,
    TCallParams=GroqCallParams,
    default_call_params=GroqCallParams(),
    setup_call=setup_call,  # type: ignore
    get_json_output=get_json_output,
    handle_stream=handle_stream,
    handle_stream_async=handle_stream_async,
)
"""A decorator for calling the Groq API with a typed function.

This decorator is used to wrap a typed function that calls the Groq API. It parses
the docstring of the wrapped function as the messages array and templates the input
arguments for the function into each message's template.

Example:

```python
from mirascope.core import prompt_template
from mirascope.core.groq import groq_call


@groq_call("llama-3.1-8b-instant")
@prompt_template("Recommend a {genre} book")
def recommend_book(genre: str):
    ...

response = recommend_book("fantasy")
print(response.content)
```

Args:
    model: The Groq model to use in the API call.
    stream: Whether to stream the response from the API call.
    tools: The tools to use in the Groq API call.
    response_model: The response model into which the response should be structured.
    output_parser: A function for parsing the call response whose value will be returned
        in place of the original call response.
    json_mode: Whether to use JSON Mode.
    client: An optional custom client to use in place of the default client.
    **call_params: The `GroqCallParams` call parameters to use in the API call.

Returns:
    The decorator for turning a typed function into an Groq API call.
"""
