![Header image](assets/Header.png)

# Agentica Python SDK

[![PyPI version](https://img.shields.io/pypi/v/symbolica-agentica.svg)](https://pypi.org/project/symbolica-agentica/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1470799122717085941?logo=discord&label=Discord)](https://discord.gg/bddGs8bb)
[![Twitter](https://img.shields.io/twitter/follow/symbolica?style=flat&logo=x&label=Follow)](https://x.com/symbolica)

[Agentica](https://agentica.symbolica.ai) is a type-safe AI framework that lets LLM agents integrate with your code—functions, classes, live objects, even entire SDKs. Instead of building MCP wrappers or brittle schemas, you pass references directly; the framework enforces your types at runtime, constrains return types, and manages agent lifecycle.

## Documentation

The full documentation can be found at [docs.symbolica.ai](https://docs.symbolica.ai).

## Prerequisites
- Python ≥ 3.12.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Quick Start

There are two ways to use the Agentica framework. You may install the package from [pypi](https://pypi.org/project/symbolica-agentica/) and sign up for an account on the [Symbolica Platform](https://www.symbolica.ai/login) to recieve $50 in free inference credits. This is the easiest way to get started.

Alternatively if you wish to develop locally or BYOK you may set up and run a local copy of the server and framework. This requires more setup and is harder to get started.

To use the platform, get an API key from [here](https://www.symbolica.ai/login).

<details>
<summary><big><b>To BYOK follow these instructions</b></big></summary>
<br>

**1. Clone the SDK**

```bash
git clone https://github.com/symbolica-ai/agentica-python-sdk
```

**2. Clone and sync the [agentica-server](https://github.com/symbolica-ai/agentica-server)** (in one terminal):

```bash
git clone https://github.com/symbolica-ai/agentica-server
cd agentica-server && uv sync
```

**3. Set up your keys, inference endpoint, and run the server**

```bash
export INFERENCE_API_KEY=<your-key>
export INFERENCE_ENDPOINT_URL="https://openrouter.ai/api/v1/chat/completions"
# export INFERENCE_ENDPOINT_URL="https://api.openai.com/v1/responses"
# export INFERENCE_ENDPOINT_URL="https://api.anthropic.com/v1/responses"
uv run src/application/main.py \
  --inference-token=$INFERENCE_API_KEY \
  --inference-endpoint $INFERENCE_ENDPOINT_URL
```

</details>

---

### Run the chat demo

The `agentica-chat` demo lets you chat with your Python REPL using the Agentica framework. You can use natural langauge to interact with objects, write and run code, and more.

#### Make the UV project

```bash
uv init agentica-chat
cd agentica-chat ; uv sync
```

#### If you are running on the Symbolica Platform

```bash
uv pip install symbolica-agentica
```

#### Or if you are running locally

```bash
uv add ../agentica-python-sdk
```

#### Create main.py

```python
import asyncio
from agentica import spawn
from agentica.logging import set_default_agent_listener
from agentica.logging.loggers import StreamLogger

set_default_agent_listener(None)

# Define your tools here!
def my_tool(arg: str) -> str:
    """Describe what your tool does."""
    return f"Result for: {arg}"

async def main():
    agent = await spawn(premise="You are a helpful assistant.")

    print("Agent ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Stream the agent's response
        stream = StreamLogger()
        with stream:
            result = asyncio.create_task(
                agent.call(str, user_input, my_tool=my_tool)  # Pass your tools here
            )

        async for chunk in stream:
            if chunk.role == "agent":
                print(chunk, end="", flush=True)
        print()

        print(f"Agent: {await result}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

#### If you are running on the Symbolica Platform

```bash
export AGENTICA_API_KEY=<your-platform-api-key>
uv run python main.py
```

#### Or if you are running locally

```bash
export S_M_BASE_URL=http://localhost:2345
uv run python main.py
```

## License

This project is licensed under the [MIT License](https://github.com/symbolica-ai/agentica-python-sdk/blob/main/LICENSE).

