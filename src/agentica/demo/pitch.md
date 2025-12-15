# Introducing Agentica

Code is the most expressive interface through which models can interact with their environment.

This observation transforms both what agents can accomplish and how developers build with them.

## Why Code Over JSON?

The industry is realizing that calling tools via code functions beats writing JSON objects. Here's the difference:

**JSON mode** — verbose, repetitive:
```json
[
  { "tool": "weather", "args": { "location": "San Francisco", "date": "2025-11-01" } },
  { "tool": "weather", "args": { "location": "San Francisco", "date": "2025-11-02" } },
  ...
]
```

**Code mode** — expressive, compact:
```python
[get_weather("San Francisco", f"2025-11-{day}") for day in range(1, 13)]
```

Two key advantages:
1. Intermediate results flow by reference instead of repetition
2. Language primitives (iteration, conditionals) that models already understand

But calling MCP tools in code is just the beginning.

## Going Beyond MCP

The status quo reduces codebases to flat functions with serializable data. That's not how developers work.

### Code Structure Informs Agents

Methods return objects with rich structure and live methods. Call `database.get_user(id)` and you receive a `User` with `update()`, `delete()`, `get_posts()`. Available operations emerge from the data itself.

```python
class User:
    def update(self) -> None: ...
    def delete(self) -> None: ...
    def get_posts(self) -> list[Post]: ...

class Database:
    def get_user(self, user_id: str) -> User: ...
    def create_user(self, name: str, posts: list[Post]) -> User: ...
```

### Accumulating Scope

Rather than a static toolbox, agents have a **scope** — a dynamic collection of objects that evolves as they work. Define the initial scope, and capabilities are discovered through methods those objects expose.

```python
database = Database(...)
agent = await spawn("You are a helpful assistant.", database)
```

The structure of your organized codebase becomes context. Methods required to be called in order? The type system communicates that naturally — no more begging agents "DO NOT CALL X UNTIL YOU HAVE CALLED Y".

### Agents Engineer Their Own Context

Instead of forcing everything through the context window, agents engineer their workspace: extracting what matters, ignoring what doesn't. Regex through a large prompt, or dispatch sub-agents to analyze subsections of tabular data.

### Giving Agents Intellisense

Developers don't read entire codebases upfront — they explore via IDE features. Agentica provides `show_definition`, letting agents inspect any object on demand:

```python
price_builder = market_data.price(symbol="XAU/USD")
show_definition(price_builder)  # agent sees available methods
```

```
class PriceEndpoint:
    def as_csv(self, **kwargs): ...
    def as_json(self): ...
    def as_pandas(self, **kwargs): ...
```

This keeps unnecessary information out of context while giving access to everything needed.

### Stop Wrapping, Start Importing

Libraries with good docstrings and type annotations work directly — no wrapper tools needed:

```python
from twelvedata import TDClient

market_data = TDClient(os.getenv("TWELVE_DATA_API_KEY"))
agent = await spawn("You are a financial analysis assistant", { 'market_data': market_data })
```

**Anything compatible with Python or TypeScript is compatible with Agentica.**

### Typed Outputs Enable Composition

Specify return types, and the execution environment won't terminate until the agent constructs that type or raises an exception:

```python
agent = await spawn("You are a helpful assistant.", { "database": database })
new_user = await agent.call(User, "Add a new random user to the database")
await new_user.onboard()  # guaranteed to be User
```

### Dynamic Multi-Agent Orchestration

Give agents the Agentica SDK itself:

```python
agent = await spawn(
    "You are an orchestrator.",
    { "spawn_agent": spawn, "database": database }
)

await agent.call(dict[UserId, str], "For each user, summarise their spending habits.")
```

The orchestrator spawns sub-agents, parallelizes work, and returns well-typed results.

## Production Architecture

Arbitrary code execution requires security guarantees:
- Agents can only use the tools they've been given
- Agents can't read or leak your codebase
- Information cannot leak between agents
- Agents can't damage tools, servers, or infrastructure

### Remote Object Proxying with Warp

**Warp** lets agents interact with objects across network boundaries as though they were local, while constraining them to well-typed operations.

Requirements we built Warp to handle:
- Async execution of async functions, methods, and futures/promises (RPC)
- Runtime type information
- Language-agnostic object model (operate TypeScript objects from Python)
- Deep object fakes for seamless agent experience

### Defense in Depth: MicroVMs + WASM

Two layers of sandboxing protect your deployment:

**MicroVMs** — Groups of agents get a full microVM, the gold standard for arbitrary code execution. Fresh OS with its own kernel, booted in seconds.

**WASM** — Per-agent isolation where performance matters. Protects filesystem and network from agent code. Each agent gets a pristine execution environment.

This architecture requires zero code changes — it's the default when you supply an API key.
