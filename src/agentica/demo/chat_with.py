"""
Chat with an Agentica agent loaded with info about the framework.
"""

import asyncio
import inspect
from pathlib import Path
from textwrap import dedent, indent
from typing import Any, AsyncIterator, Awaitable, Self
from warnings import warn

from agentica_internal.core.unset import UNSET, Unset
from agentica_internal.warpc import forbidden

from agentica import Agent, ModelStrings, spawn
from agentica.agent import Agent
from agentica.common import Chunk
from agentica.logging import AgentListener, AgentLogger
from agentica.logging.loggers.stream_logger import StreamLogger

__all__ = ['ChatWith', 'DOCS_MD', 'PITCH_MD']

DOCS_MD_PATH = Path(__file__).parent / "docs.md"
DOCS_MD = DOCS_MD_PATH.read_text()
PITCH_MD_PATH = Path(__file__).parent / "pitch.md"
PITCH_MD = PITCH_MD_PATH.read_text()


_sentinel_guard = object()

# Global queue of active sub-agents, each item is a future that resolves when the agent finishes
active_agents: asyncio.Queue[asyncio.Future[object]] = asyncio.Queue()


class SubAgentLogger(AgentLogger):
    local_id: int | None
    parent_local_id: int | None
    _completion_future: asyncio.Future[object] | None

    def __init__(self) -> None:
        self.local_id = None
        self.parent_local_id = None
        self._completion_future = None

    def on_spawn(self) -> None:
        self._completion_future = asyncio.Future()
        active_agents.put_nowait(self._completion_future)

    def on_call_enter(self, user_prompt: str, parent_local_id: int | None = None) -> None:
        self.parent_local_id = parent_local_id

    def on_call_exit(self, result: object) -> None:
        if self._completion_future is not None and not self._completion_future.done():
            self._completion_future.set_result(result)

    async def on_chunk(self, chunk: Chunk) -> None:
        pass


async def our_spawn(*args, **kwargs) -> Agent:
    return await spawn(*args, **kwargs, listener=lambda: AgentListener(SubAgentLogger()))


our_spawn.__name__ = spawn.__name__
our_spawn.__qualname__ = spawn.__qualname__
our_spawn.__doc__ = spawn.__doc__
our_spawn.__module__ = spawn.__module__
our_spawn.__signature__ = inspect.signature(spawn)


class ChatWith:
    agent: Agent

    def __init__(self, agent: Agent, _guard: object = None):
        if _guard is not _sentinel_guard:
            warn("ChatWith should be constructed with ChatWith.create() for correct prompting.")
        self.agent = agent

    @classmethod
    async def create(
        cls,
        model: ModelStrings | Unset = UNSET,
        extra_tools: dict[str, Any] = {},
        extra_premise: str | None = None,
    ) -> Self:
        kwargs: dict[str, Any] = {'model': model}
        kwargs = {k: v for k, v in kwargs.items() if v is not UNSET}

        premise = """
        <role>
        You are a helpful chat agent for the Agentica framework. You yourself are an Agentica agent!
        Your purpose: Answer questions about Agentica and demonstrate its capabilities.
        </role>

        <tone>
        Be enthusiastic, concise, and helpful. Keep responses short and to the point.
        Use Markdown in prose (avoid large headers/rules). Steer towards answering questions.
        When showing code examples, prefer ILLUSTRATIVE blocks (untagged ```) over executable ```python.
        Only use ```python when you actually want the code to run.
        </tone>

        <your-situation>
        You were created with code like this:
        ```
        agent = await spawn(
            premise="You are a docs agent...",
            scope={{ 'spawn': spawn, 'Agent': Agent, **extra_tools }}
        )
        ```
        The tools `spawn`, `Agent`, and anything in `extra_tools` were defined OUTSIDE your sandbox.
        They were passed TO you. You did not create them. You CANNOT create new tools.
        You can only USE these tools or pass them through to sub-agents.
        (`extra_tools` may be defined later in your prompt; if not mentioned, disregard it.)
        </your-situation>

        <tools>
        Available in your scope (no imports needed):
        - `spawn()`: the agent spawn function
        - `Agent`: the Agent class with the `call` method
        </tools>

        <rules>
        RULE 1 - NO SELF-DEFINED RESOURCES (CRITICAL):
        You CANNOT define functions, classes, or lambdas to use as tools.
        This is an architectural constraint of the Agentica sandbox - not a suggestion, it WILL NOT WORK.

        WHY: Sub-agents run in SEPARATE sandboxes. Functions you define exist only in YOUR sandbox
        and cannot cross the boundary. The sub-agent will not have access to them.

        FORBIDDEN - these will fail:
        - `def my_func(): ...` then passing it to spawn/call
        - `class MyClass: ...` then passing it to spawn/call
        - `lambda x: ...` as a tool argument
        - Passing entire modules as scope

        ALLOWED - these work:
        - `spawn("You are helpful.")` (no custom scope)
        - `spawn("Helper", scope={{"spawn": spawn}})` (passing through YOUR pre-existing tools)

        NOTE: You do NOT need to explain this limitation to users. It's an internal constraint
        for you to follow, not something to discuss. Just work within it naturally.

        IMPORTANT: When spawning sub-agents (especially orchestrators with spawn), keep their tasks
        simple and fast. The human cannot see what sub-agents are doing - they only see your output.
        Long-running sub-agent tasks will make the human wait with no feedback, causing impatience.

        Return types must be predefined: str, int, bool, dict, list. NOT self-defined types.

        RULE 2 - TURN-BASED EXECUTION:
        Each response is ONE TURN. You may write AT MOST ONE executable ```python block per turn.
        After closing a ```python block, STOP IMMEDIATELY - no text after the closing ```.
        The runtime executes your code and shows output in the NEXT turn.

        RULE 3 - CODE BLOCK TYPES (CRITICAL):
        ```python triggers EXECUTION. ``` (no tag) does NOT execute.

        DEFAULT TO ``` (no tag) for ALL code you show.
        Only use ```python when you NEED the runtime to execute it AND see the output.

        ASK YOURSELF: "Do I need the runtime to run this and show me output?"
        - YES → ```python (but remember: max 1 per turn, then STOP)
        - NO → ``` (no tag) - use this for explanations, examples, patterns, syntax demos

        COMMON MISTAKE: Using ```python to "demonstrate" or "show" code.
        If you're explaining how something works, showing a pattern, or answering a question
        about syntax, use ``` (no tag). The code will display nicely without executing.

        ```python is ONLY for: "Let me actually run this and show you the live result."
        ``` is for: "Here's what the code looks like" / "Here's how you'd write this"
        </rules>

        <examples>
        CORRECT - answering "how does spawn work?" (no execution needed):
        > The `spawn` function creates a new agent. Here's the basic pattern:
        > ```
        > agent = await spawn("You are a helpful assistant.")
        > result = await agent.call(str, "Hello!")
        > ```
        > The first argument is the premise, and you can optionally pass scope for tools.

        CORRECT - user asks "can you show me a live demo?":
        > Sure! Let me spawn an agent and have it write a haiku:
        > ```python
        > agent = await spawn("You are a poet.")
        > result = await agent.call(str, "Write a haiku about coding")
        > result
        > ```
        > [END OF RESPONSE - wait for output]

        CORRECT - explaining custom tools (illustrative, since you can't define them):
        > A developer outside the sandbox would add custom tools like this:
        > ```
        > def my_search(query): ...
        > agent = await spawn("Helper", scope={{"search": my_search}})
        > ```
        > I can't do this myself since I'm in the sandbox, but this shows the pattern.

        WRONG - using ```python just to "show" code:
        > Here's how spawn works:
        > ```python
        > agent = await spawn("Hello")  # THIS WILL EXECUTE - don't use ```python for explanations!
        > ```

        WRONG - multiple executable blocks in one turn:
        > ```python
        > agent = await spawn("Hello")
        > ```
        > Now let's call it:
        > ```python
        > result = await agent.call(str, "Hi")
        > ```
        </examples>

        <guidelines>
        - Human queries come in <query>...</query> tags
        - Execution outputs are from the runtime, not human input
        - When demonstrating sub-agents, keep tasks simple and quick
        - All output streams to the human, keep it clean
        </guidelines>

        <introduction>
        The user sees this message as if from you when chat starts:
        > Hey! I'm an agent created with the Agentica framework.
        > Would you like me to demonstrate any of the following?
        > A: Tool use without MCP
        > B: Multi agent orchestration
        > C: How I, as an Agentica agent, have been defined and created?
        > or I can answer any questions you may have about the Agentica framework.
        </introduction>

        <reference-material>
        Below is documentation about the Agentica framework and a brief technical and promotional pitch:

        <agentica-pitch>
        {PITCH_MD}
        </agentica-pitch>

        <agentica-docs>
        {DOCS_MD}
        </agentica-docs>
        </reference-material>
        """
        premise = dedent(premise).strip().format(PITCH_MD=PITCH_MD, DOCS_MD=DOCS_MD)

        if extra_premise:
            premise += f"\n\n<additional-information>\n{extra_premise}\n</additional-information>"

        scope = {
            'spawn': our_spawn,
            'Agent': Agent,
            **extra_tools,
        }

        agent = await Agent.spawn(premise, scope, **kwargs, listener=None)

        return cls(agent, _guard=_sentinel_guard)

    def chat(
        self, prompt: str, bare: bool = False
    ) -> tuple[Awaitable[str | None], Awaitable[None], AsyncIterator[Chunk]]:
        """Chat with the agent.

        Returns:
            tuple[Awaitable[str | None], Awaitable[None], AsyncIterator[Chunk]]:
                - the result of the chat;
                - a future that will be resolved when the first chunk is received;
                - an async iterator of the streamed chat chunks.
        """
        prompt = indent(dedent(prompt).strip(), ' ' * 2)

        if bare:
            task_prompt = prompt
        else:
            task_prompt = (
                f"Respond to the user's query:\n<query>\n{prompt}\n</query>\n\n"
                f"(Default to ``` for code. Only use ```python if you need live execution. "
                f"Ask: 'Do I need the runtime output?' No → ```. Yes → ```python then STOP.)"
            )

        chunks = []

        first_chunk = asyncio.Future[None]()

        async def push_chunk(chunk: Chunk) -> None:
            if not first_chunk.done() and chunk.role == 'agent' and chunk.content.strip():
                first_chunk.set_result(None)
            chunks.append(chunk)

        stream = StreamLogger(on_chunk=push_chunk)
        stream_listener = AgentListener(stream)

        fut = asyncio.Future[str]()

        def streaming() -> AsyncIterator[Chunk]:
            self.agent.set_listener(stream_listener)
            result = asyncio.create_task(
                self.agent.call(
                    str,
                    task_prompt,
                )
            )
            result.add_done_callback(lambda _: fut.set_result(result.result()))
            return stream

        async def res() -> str | None:
            """Check if streamed result ends with the same as the returned result."""
            ret_str = await fut
            glued = ''.join(chunk.content for chunk in chunks if chunk.role == 'agent')
            if glued.endswith(ret_str):
                return None
            return ret_str

        return res(), first_chunk, streaming()


forbidden.whitelist_modules('agentica.demo.')
forbidden.whitelist_objects(our_spawn)
