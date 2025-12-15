# Agentica Documentation

## Overview

Agentica is a library for integrating AI features into Python applications.

## Basics

A method of interaction is through the `@agentic()` decorator, which works like this:

```
from agentica import agentic

# Defines an AI-powered function
@agentic()
async def add(a: int, b: int) -> int:
	"""Returns the sum of a and b"""
	...

# Calls the AI-powered function
result = await add(1, 2) # This addition is done by an LLM via the agentica framework
assert result == 3
```

This allows you to use AI to implement functions which are not possible to implement in pure python. Functions decorated with `@agentic` can be `async` or sync.

The alternative syntax is to `spawn` an agent.

```
from agentica import spawn

# Defines the AI-powered agent
agent = await spawn(premise="You are a truth-teller.")

# Calls the AI-powered agent
result: bool = await agent.call(bool, "The Earth is flat")
assert result == False
```

The creation and call of agents created with `spawn` are always awaitable. When calling an agent you **must** always pass in the return type as the first argument, followed by an argument of type `str`.

### Return Types

Return types are optional and flexible:

```
# Return type defaults to str if not specified
result = await agent.call("What is 2+2?")  # Returns str

# Specify exact types for structured output
result: int = await agent.call(int, "What is 2+2?")  # Returns int
result: dict[str, int] = await agent.call(dict[str, int], "Count items by category")

# Use None for side-effects only
await agent.call(None, "Send a message to John")  # No return value needed
```

## Agent Instantiation

There are two ways to create agents:

### Using `spawn` (async)
Use `spawn` for most cases - it's awaitable and async-friendly:

```
agent = await spawn(premise="You are a helpful assistant.")
```

### Using `Agent()` directly (sync)
Use direct instantiation when you need synchronous creation, such as in `__init__` methods:

```
from agentica import Agent

class CustomAgent:
    def __init__(self, directory: str):
        # Must be synchronous - use Agent() not spawn()
        self._brain = Agent(
            premise="You are a specialized assistant.",
            scope={"tool": some_tool}
        )

    async def run(self, task: str) -> str:
        return await self._brain(str, task)
```

**Tip**: Direct `Agent` instantiation is particularly useful when building custom agent classes or in contexts that cannot be async.

## Premise vs System Prompt

You can control the agent's instructions in two ways:

```
# Use 'premise' to add context to the default system prompt
agent = await spawn(premise="You are a math expert.")

# Use 'system' for full control of the system prompt
agent = await spawn(system="You are an AI assistant. Always respond with JSON.")
```

**Note**: You cannot use both `premise` and `system` together.

## Passing in objects

If you want the AI-powered function or agent to use a function, class, object etc. inside an agentic function, simply put it in the `@agentic` decorator or `scope` in the call to `spawn`.

```
from agentica import agentic, spawn

# User-defined function
from tools import web_search

# Defines the AI-powered agent
agent = await spawn(premise="You are a truth-teller.", scope={"web_search" : web_search})

# Defines the AI-powered function
@agentic(web_search)
def truth_teller(statement: str) -> bool:
  """Returns whether or not a statement is True or False."""
  ...
```

### SDK Integration Pattern

Extract specific methods from SDK clients for focused scope:

```
from slack_sdk import WebClient

# Extract only the methods you need
slack_conn = WebClient(token=SLACK_BOT_TOKEN)
list_users = slack_conn.users_list
send_message = slack_conn.chat_postMessage

@agentic(list_users, send_message, model="openai:gpt-5")
async def send_team_update(message: str) -> None:
    """Send a message to all team members."""
    ...
```

### Per-Call Scope

You can also add scope per invocation:

```
agent = await spawn(premise="Data analyzer")

# Add resources for this specific call
result = await agent.call(
    dict[str, int],
    "Analyze the dataset",
    dataset=pd.read_csv("data.csv").to_dict(),
    analyzer_tool=custom_analyzer
)
```

## Model Selection

Agentica supports multiple AI models. Specify with the `model` parameter:

```
# For agents
agent = await spawn(
    premise="Fast responses needed",
    model="openai:gpt-5"  # Default is 'openai:gpt-4.1'
)

# For agentic functions
@agentic(model="anthropic:claude-sonnet-4.5")
async def analyze(text: str) -> dict:
    """Analyze the text."""
    ...
```

**Supported models**:
- `openai:gpt-3.5-turbo`
- `openai:gpt-4o`
- `openai:gpt-4.1` (default)
- `openai:gpt-5`
- `anthropic:claude-sonnet-4`
- `anthropic:claude-opus-4.1`
- `anthropic:claude-sonnet-4.5`
- `anthropic:claude-opus-4.5`

or any OpenRouter model, by providing the OpenRouter canonical model slug.

## Token Limits

Control the maximum number of tokens generated in a single inference round with `max_tokens`:

```
# For agents
agent = await spawn(
    premise="Brief responses only",
    max_tokens=500  # Limit response length
)

# For agentic functions
@agentic(max_tokens=1000)
async def summarize(text: str) -> str:
    """Create a concise summary."""
    ...
```

**Use cases**:
- Ensure brief responses for cost control
- Prevent overly long outputs
- Match specific output length requirements

If the response would exceed `max_tokens`, a `MaxTokensError` will be raised. See [Error Handling](#error-handling) for how to handle this.

## Persistence

Agentic functions can maintain state between calls:

```
# Stateful agentic function
@agentic(persist=True, model="openai:gpt-4.1")
async def chatbot(message: str) -> str:
    """A chatbot that remembers conversation history."""
    ...

# First call
response1 = await chatbot("My name is Alice")

# Second call - remembers previous context
response2 = await chatbot("What's my name?")  # Will know it's Alice
```

**Tip**: Use `persist=True` when you need conversation history or stateful behavior in agentic functions. For agents, state is maintained automatically across calls to the same agent instance.

## Streaming

Streaming is supported for both agents and agentic function, most straightforwardly by using a `StreamLogger`.

```
from agentica import spawn
from agentica.logging.loggers import StreamLogger

agent = await spawn(premise="You are a truth-teller.")

stream = StreamLogger()
with stream:
    result = asyncio.create_task(
        agent.call(bool, "Is Paris the capital of France?")
    )
```

This creates an async generator `stream` containing producing the streamed in text-chunks of the invocation within the with-statement.
The stream can be consumed like this:

```
# Consume stream FIRST for live output
async for chunk in stream:
    print(chunk.content, end="", flush=True)

# Then await result
final_result = await result
```

**Important**: The stream should be consumed **before** awaiting the result, otherwise you won't see the live text generation.

## Logging and Debugging

Agentica provides built-in logging to help debug agents and agentic functions.

### Default Logging

By default, all agents and agentic functions use `StandardListener` which:
- Prints lifecycle events to stdout with colors
- Writes chat histories to `./logs/agent-<id>.log`

```shell
Spawned Agent 25 (./logs/agent-25.log)
► Agent 25: Calculate the 32nd power of 3
◄ Agent 25: 1853020188851841
```

### Contextual Logging

Temporarily change logging for specific code sections:

```
from agentica.logging.loggers import FileLogger, PrintLogger

# Only log to file in this block
with FileLogger():
    agent = await spawn(premise="Debug this agent")
    await agent.call(int, "Calculate something")

# Multiple loggers can be nested
with PrintLogger():
    with FileLogger():
        # Both print AND file logging active
        agent = await spawn(premise="Dual logging")
        await agent.call(str, "Hello")
```

### Disable Logging

```
from agentica.logging.agent_logger import NoLogging

with NoLogging():
    agent = await spawn(premise="Silent agent")
    await agent.call(int, "Secret calculation")
```

### Per-Agent/Function Logging

```
from agentica.logging import PrintOnlyListener, FileOnlyListener

# Attach listener to specific agent
agent = await spawn(
    premise="Custom logging",
    listener=PrintOnlyListener  # Only print, no files
)

# Attach listener to specific agentic function
@agentic(listener=FileOnlyListener)
async def my_func(a: int) -> str:
    """Only logs to file."""
    ...
```

### Global Logging Configuration

```
from agentica.logging import set_default_agent_listener, PrintOnlyListener

# Change default for all agents/functions
set_default_agent_listener(PrintOnlyListener)

# Disable all logging by default
set_default_agent_listener(None)
```

**Logging Priority** (highest to lowest):
1. Contextual loggers (via `with` statement)
2. Per-agent/function listener
3. Default listener

## Using MCP

If you want the AI-powered function or agent to use an MCP server, simply put it in the `@agentic` decorator or the call to `spawn`.

```
from agentica import agentic, spawn

# Defines the AI-powered agent
agent = await spawn(premise="You are a truth-teller.", mcp="path/to/config.json")

# Defines the AI-powered function
@agentic(mcp="path/to/config.json")
def truth_teller(statement: str) -> bool:
  """Returns whether or not a statement is True or False."""
  ...
```

where "path/to/config.json" is a standard JSON config file such as:

```json
{
  "mcpServers": {
    "tavily-remote-mcp": {
      "command": "npx -y mcp-remote https://mcp.tavily.com/mcp/?tavilyApiKey=<your-api-key>",
      "env": {}
    }
  }
}
```

## Error Handling

Agentica provides comprehensive error handling through the `agentica.errors` module.

### SDK Errors

All Agentica errors inherit from `AgenticaError`, making it easy to catch all SDK-related errors:

```
from agentica import agentic
from agentica.errors import AgenticaError, RateLimitError, InferenceError

@agentic()
async def process_data(data: str) -> dict:
    """Process the data."""
    ...

try:
    result = await process_data(raw_data)
except RateLimitError as e:
    # Handle rate limiting
    await asyncio.sleep(60)
    result = await process_data(raw_data)
except InferenceError as e:
    # Handle all inference service errors
    logger.error(f"Inference failed: {e}")
    result = {}
except AgenticaError as e:
    # Catch any other SDK errors
    logger.error(f"Agentica error: {e}")
    raise
```

**Error hierarchy:**
- `AgenticaError` - Base for all SDK errors
  - `ServerError` - Base for remote operation errors
    - `GenerationError` - Base for agent generation errors
      - `InferenceError` - HTTP errors from inference service
      - `MaxTokensError`, `ContentFilteringError`, etc.
  - `ConnectionError` - WebSocket and connection errors
  - `InvocationError` - Agent invocation errors

### Custom Exceptions

You can define custom exceptions and pass them into the `@agentic()` decorator so the agent can raise them:

```
class DataValidationError(Exception):
    """Raised when input data fails validation."""
    pass

@agentic(DataValidationError)
async def analyze_data(data: str) -> dict:
    """
    Analyze the dataset.

    Raises:
        DataValidationError: If data is empty or malformed
        ValueError: If data format is not supported

    Returns a dictionary with analysis results.
    """
    ...

try:
    result = await analyze_data(raw_data)
except DataValidationError as e:
    logger.warning(f"Invalid data: {e}")
    result = {"status": "validation_failed"}
```

**Tip**: The agent can see your docstrings! Document exception conditions clearly in the `Raises:` section, and the agent will raise them appropriately.

## Common Patterns

### Stateful Data Analysis

Agents maintain context across calls and can manipulate variables by reference:

```
from agentica import spawn
import pandas as pd

agent = await spawn()

# First analysis
result = await agent.call(
    dict[str, int],
    "Count movies by genre",
    dataset=pd.read_csv("movies.csv").to_dict()
)

# Agent remembers previous result
filtered = await agent.call(
    dict[str, int],
    "Keep only genres with more than 1000 movies"
)
```

### Custom Agent Classes

Wrap `Agent` for domain-specific functionality:

```
from agentica.agent import Agent

class ResearchAgent:
    def __init__(self, web_search_fn):
        self._brain = Agent(
            premise="You are a research assistant.",
            scope={"web_search": web_search_fn}
        )

    async def research(self, topic: str) -> str:
        return await self._brain(str, f"Research: {topic}")

    async def summarize(self, text: str) -> str:
        return await self._brain(str, f"Summarize: {text}")

# Use it
researcher = ResearchAgent(web_search)
findings = await researcher.research("AI agents in 2025")
summary = await researcher.summarize(findings)
```

### Multi-Agent Orchestration

Coordinate multiple agents for complex tasks:

```
from agentica.agent import Agent

class LeadResearcher:
    def __init__(self):
        self._brain = Agent(
            premise="Coordinate research tasks across subagents.",
            scope={"SubAgent": ResearchAgent}
        )

    async def call(self, query: str) -> str:
        return await self._brain.call(str, query)

# The lead researcher can spawn and coordinate subagents
lead = LeadResearcher()
report = await lead("Research companies building AI agents")
```