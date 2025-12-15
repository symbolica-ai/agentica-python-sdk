r"""

- `CaptionFormatter`: responsible for formatting/printing captions.
- `Captioner`: does captioning logic, iterates, attaches to formatter.
- `CaptionLogger`: AgentLogger which pumps chunks to Captioner.
- `CaptionedAgent`: looks like an agent (wraps one) but does __call__ and echo through a CaptionLogger.
- `caption()`: wraps an agent in CaptionedAgent or decorates an agentic function so all calls go through CaptionLogger.

```
agent1 = spawn(...)
agent2 = spawn(...)

# Without this will use global caption formatter.
with CaptionLogger():
    asyncio.gather(
        agent1("Do task 1"),
        agent2("Do task 2"),
    )
```
prints animated blocks like so:
```
[/] Doing task 1...
[\] Doing task 2...
```

"""

import asyncio
import re
from abc import ABC, abstractmethod
from textwrap import dedent
from types import FunctionType
from typing import Any, Callable, overload, override

from agentica_internal.core.ansi import palettes
from agentica_internal.core.log import LogBase
from agentica_internal.core.utils import copy_signature

from agentica.agent import Agent
from agentica.common import ModelStrings
from agentica.errors import MaxTokensError
from agentica.logging.agent_listener import NoopListener
from agentica.logging.agent_logger import DefaultLogId, LogId, NamedLogId, NoLogging
from agentica.logging.loggers import StreamLogger
from agentica.std.utils import spoof_signature
from agentica.template import template


class CaptionFormatter(ABC):
    """Configurable formatter/printer for captions."""

    @abstractmethod
    async def send(self, local_id: LogId, summary: str) -> None:
        """Take a summary and do something with it."""
        ...

    @abstractmethod
    def enter(self, local_id: LogId) -> None:
        """Signal that an agent is starting."""
        ...

    @abstractmethod
    def done(self, local_id: LogId) -> None:
        """Signal that an agent is done."""
        ...


class CaptionPrintFormatter(CaptionFormatter):
    """Prints captions to stdout."""

    summaries: dict[LogId, str]
    _task: asyncio.Task[None] | None
    _stop_event: asyncio.Event
    _refresh_per_second: float

    def __init__(self) -> None:
        super().__init__()
        self.summaries = dict()
        self._task = None
        self._stop_event = asyncio.Event()
        self._refresh_per_second = 8.0

    @override
    async def send(self, local_id: LogId, summary: str) -> None:
        # Some truncation. Should ideally not be necessary.
        MAX_LEN = 100
        SUFFIX = ' [...]'
        summary = summary.strip()
        if '\n' in summary:
            summary = summary.split('\n')[0]
        if len(summary) > MAX_LEN:
            summary = summary[: MAX_LEN - len(SUFFIX)] + SUFFIX

        self.summaries[local_id] = summary

        # Ensure the live printer is running.
        self._ensure_started()

    @override
    def enter(self, local_id: LogId) -> None:
        self.summaries[local_id] = ''
        self._ensure_started()

    @override
    def done(self, local_id: LogId) -> None:
        # Remove a task line; if none remain, stop the live worker and clear UI
        _ = self.summaries.pop(local_id, None)

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            # Reset stop event in case we were stopped before
            if self._stop_event.is_set():
                self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(self._run_live())

    async def _run_live(self) -> None:
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.table import Table

        def render_table_from_items(items: list[tuple[LogId, str]]) -> Table:
            table = Table.grid(padding=(0, 1))
            table.add_column(no_wrap=True)
            table.add_column()
            for _local_id, _summary in sorted(items, key=lambda kv: str(kv[0])):
                display = _summary
                if not display:
                    prefix = f"Agent {_local_id}. "
                else:
                    prefix = f"Agent {_local_id}: "
                table.add_row(
                    Spinner("point", style=f"bold {palettes.Agentica.periwinkle.hex_str}"),
                    f"[bold]{prefix}[/][grey53]{display}[/]",
                )
            return table

        try:
            # Start with an empty renderable to enter Live context
            with Live(
                render_table_from_items([]),
                refresh_per_second=self._refresh_per_second,
                transient=True,
            ) as live:
                # Refresh at a steady cadence so spinners animate even without new updates
                sleep_s = 1.0 / self._refresh_per_second
                while not self._stop_event.is_set():
                    await asyncio.sleep(sleep_s)
                    # Snapshot under lock to avoid mutation during iteration
                    items = list(self.summaries.items())
                    if items:
                        live.update(render_table_from_items(items), refresh=False)
                    else:
                        # No more lines -> update to empty, exit loop; Live will
                        # clear the display because transient=True
                        live.update(render_table_from_items([]), refresh=False)
                        break
        finally:
            # Reset state so we can be restarted on future updates
            self._task = None
            if self._stop_event.is_set():
                self._stop_event = asyncio.Event()

    async def aclose(self) -> None:
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                pass

    def close(self) -> None:
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_default_caption_formatter = CaptionPrintFormatter()


class Captioner(LogBase):
    """Dispatches captions to captioning agent."""

    caption_formatter: CaptionFormatter
    _captioning_agents: dict[LogId, Agent]
    _per_summary_prompt: str
    _ignore_keyword: str
    _system: str
    _model: ModelStrings
    _every_n: int
    _locks: dict[LogId, asyncio.Lock]
    _stops: dict[LogId, asyncio.Event]

    def __init__(
        self,
        *,
        caption_formatter: CaptionFormatter,
        system: str | None = None,
        model: ModelStrings = 'openai/gpt-4o-mini:nitro',  # gpt-3.5-turbo is too weak
        per_summary_prompt: str | None = None,
        ignore_keyword: str | None = None,
        every_n_tokens: int = 100,
        logging: bool = False,
    ) -> None:
        super().__init__(logging=logging)
        self.caption_formatter = caption_formatter
        self._captioning_agents = {}
        self._locks = {}
        self._stops = {}

        # Either provide all prompt pieces (system/per_summary_prompt/ignore_keyword),
        # or let us use all defaults. Mixing defaults & custom pieces makes the
        # interaction brittle and hard to reason about.
        provided = (system is not None, per_summary_prompt is not None, ignore_keyword is not None)
        if any(provided) and not all(provided):
            raise ValueError("system, per_summary_prompt, and ignore_keyword must be set in tandem")

        if system is None:
            # Show concrete input→output transformations. Models learn from examples.
            system = """You write short status updates for a progress indicator.

INPUT: "Let me search for information about Python decorators..."
OUTPUT: Searching for Python decorator info

INPUT: "```python\ndef hello():\n    print('world')\n```"
OUTPUT: Writing Python function

INPUT: "The results show that option A costs $50 and option B costs $75..."
OUTPUT: Comparing pricing options

INPUT: "I'll now analyze the data from the API response..."
OUTPUT: Analyzing API response data

INPUT: "Based on my previous analysis, I recommend..."
OUTPUT: Formulating recommendations

Rules:
- Output 3-10 words starting with -ing verb
- Output NO_UPDATE if the activity hasn't changed
- Never start with "I" or "The agent"
- Never give advice or ask questions
- Never output code or lists"""
            system = dedent(system).strip()

        # Store config; create a dedicated captioner agent per local_id on demand
        self._system = system
        self._model = model

        if per_summary_prompt is None:
            # Match the INPUT/OUTPUT pattern from system prompt.
            per_summary_prompt = "INPUT: {agent_output}\nOUTPUT:"

        self._per_summary_prompt = per_summary_prompt

        if ignore_keyword is None:
            ignore_keyword = 'NO_UPDATE'

        self._ignore_keyword = ignore_keyword
        self._every_n = every_n_tokens

    def _sanitize_caption(self, raw: str) -> str | None:
        """
        Light cleanup only of the raw caption.
        """
        s = (raw or "").strip()
        if not s:
            return None

        # First line only, collapse whitespace.
        lines = s.splitlines()
        s = re.sub(r"\s+", " ", lines[0] if lines else "").strip()
        if not s:
            return None

        # NO_UPDATE means no change.
        if self._ignore_keyword in s.upper():
            return None

        # Strip quotes/backticks that models sometimes add.
        s = s.strip("`\"'")

        # Strip trailing punctuation.
        s = s.rstrip(".!?,;:")

        # Reasonable length limit.
        words = s.split()
        if len(words) > 12:
            s = " ".join(words[:12])

        return s if s else None

    async def consume(self, local_id: LogId, stream: StreamLogger) -> None:
        summary: str = ""
        collated = ""
        n = 1
        role = None

        try:
            async for chunk in stream:
                # If caller signaled stop, exit early without further summaries/sends
                if self._is_stopped(local_id):
                    break
                if role != chunk.role:
                    collated += f'\n\n[{chunk.role}]\n'
                    role = chunk.role

                if n % self._every_n == 0:
                    try:
                        maybe_summary = await self.make_summary(local_id, collated)
                        if maybe_summary:
                            summary = maybe_summary
                            await self.caption_formatter.send(local_id, summary)
                    except Exception as e:
                        self.dump_exception(
                            e,
                            f"Captioner failed to generate summary for local_id {local_id}",
                        )
                        try:
                            await self.caption_formatter.send(local_id, "[Caption error]")
                        except:
                            pass
                    collated = ""

                collated += chunk.content
                n += 1

            # summarize remaining tokens
            if not self._is_stopped(local_id) and len(collated.split()) > self._every_n // 2:
                try:
                    if rem := await self.make_summary(local_id, collated):
                        await self.caption_formatter.send(local_id, rem)
                except Exception as e:
                    self.dump_exception(
                        e,
                        f"Captioner failed to generate final summary for local_id {local_id}",
                    )
                    try:
                        await self.caption_formatter.send(local_id, "[Caption error]")
                    except:
                        pass
        except Exception as e:
            self.dump_exception(
                e,
                f"Captioner task encountered unexpected error for local_id {local_id}",
            )
        finally:
            # Ensure formatter/agent cleanup happens after the stream fully drains
            self.done(local_id)

    def consume_task(self, local_id: LogId, stream: StreamLogger) -> asyncio.Task[None]:
        return asyncio.create_task(self.consume(local_id, stream))

    def _get_agent(self, local_id: LogId) -> Agent:
        agent = self._captioning_agents.get(local_id)
        if agent is None:
            agent = Agent(
                system=self._system,
                model=self._model,
                listener=NoopListener,
                # Hard cap output: an 8-word caption is ~15 tokens + buffer.
                # This prevents runaway outputs even if the model ignores the prompt.
                max_tokens=30,
            )
            self._captioning_agents[local_id] = agent
        return agent

    def _get_lock(self, local_id: LogId) -> asyncio.Lock:
        lock = self._locks.get(local_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[local_id] = lock
        return lock

    def _get_stop_event(self, local_id: LogId) -> asyncio.Event:
        ev = self._stops.get(local_id)
        if ev is None:
            ev = asyncio.Event()
            self._stops[local_id] = ev
        return ev

    def _is_stopped(self, local_id: LogId) -> bool:
        ev = self._stops.get(local_id)
        return ev.is_set() if ev is not None else False

    def stop(self, local_id: LogId) -> None:
        self._get_stop_event(local_id).set()

    async def make_summary(self, local_id: LogId, collated: str) -> str | None:
        summary_raw: str = ""

        lock = self._get_lock(local_id)
        async with lock:
            # If stopped, do nothing
            if self._is_stopped(local_id):
                return None
            with NoLogging():
                collated = collated.strip()
                agent = self._get_agent(local_id)
                # The agent is persistent, so prior captions are in chat history.
                task = self._per_summary_prompt.format(agent_output=collated)

                # Retry loop: if we hit MaxTokensError, nudge the model to be shorter.
                MAX_RETRIES = 2
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        if attempt == 0:
                            summary_raw = await agent.call(str, template(task))
                        else:
                            # Retry: just ask for shorter. No echobait.
                            summary_raw = await agent.call(str, template("Shorter."))
                        break
                    except MaxTokensError:
                        if attempt == MAX_RETRIES:
                            # Give up, return nothing rather than crash.
                            return None
                        # Otherwise loop and retry with shorter prompt.
                        continue

            # Re-check stop after await
            if self._is_stopped(local_id):
                return None

        return self._sanitize_caption(summary_raw)

    def enter(self, local_id: LogId) -> None:
        self.caption_formatter.enter(local_id)

    def done(self, local_id: LogId) -> None:
        self.caption_formatter.done(local_id)
        agent = self._captioning_agents.pop(local_id, None)
        if agent is not None:
            try:
                agent.sync_close()
            except Exception:
                pass
        _ = self._locks.pop(local_id, None)
        _ = self._stops.pop(local_id, None)


class CaptionLogger(StreamLogger):
    captioner: Captioner
    _default_name: str | None
    _task: asyncio.Task[None] | None

    def __init__(
        self,
        default_name: str | None = None,
        caption_formatter: CaptionFormatter = _default_caption_formatter,
    ) -> None:
        super().__init__()
        self._default_name = default_name
        self.captioner = Captioner(caption_formatter=caption_formatter)

    @override
    def on_spawn(self) -> None:
        if self.local_id is None:
            if self._default_name is not None:
                self.local_id = NamedLogId(self._default_name)
            else:
                self.local_id = DefaultLogId()

    @override
    def on_call_enter(self, user_prompt: str, parent_local_id: LogId | None = None) -> None:
        super().on_call_enter(user_prompt, parent_local_id)
        assert self.local_id is not None
        self._task = self.captioner.consume_task(self.local_id, self)
        self.captioner.enter(self.local_id)

    @override
    def on_call_exit(self, result: object) -> None:
        assert self.local_id is not None
        # Signal stop so no more summaries are produced/sent
        self.captioner.stop(self.local_id)
        super().on_call_exit(result)
        # Do not cancel or finalize here; the stream sentinel will end the
        # consumer, which will finalize in its own finally block.

    async def __aenter__(self) -> None:
        return self.__enter__()

    @override
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Also signal stop on context exit for safety
        if self.local_id is not None:
            self.captioner.stop(self.local_id)
        e = super().__exit__(exc_type, exc_value, traceback)
        # Don't cancel the consumer here; let it drain naturally.
        # If used as an async context manager, __aexit__ will await it.
        return e

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if self.local_id is not None:
            self.captioner.stop(self.local_id)
        if self._task is not None:
            try:
                await self._task
            except Exception as e:
                self.captioner.dump_exception(e, "Caption task raised exception during cleanup")
        return self.__exit__(exc_type, exc_value, traceback)

    def __del__(self) -> None:
        # Avoid cancelling; consumer should be short-lived after stream ends.
        self._task = None


class CaptionedAgent:
    _captioned_agent: Agent
    _captione_formatter: CaptionFormatter

    def __init__(
        self,
        agent: Agent,
        *,
        caption_formatter: CaptionFormatter,
    ) -> None:
        self._captioned_agent = agent
        self._captione_formatter = caption_formatter

    @copy_signature(Agent.call)
    def __call__(self, *args, **kwargs) -> Any:
        """See ``Agent.__call__``."""
        with CaptionLogger(caption_formatter=self._captione_formatter):
            return self._captioned_agent.call(*args, **kwargs)


@overload
def caption(
    agent: Agent,
    /,
    caption_formatter: CaptionFormatter,
) -> CaptionedAgent: ...


@overload
def caption[F: FunctionType](
    agentic_function: Callable[[F], F],  # noqa: F841
    /,
    caption_formatter: CaptionFormatter,
) -> Callable[[F], F]: ...


def caption[F: FunctionType](
    magician: Agent | Callable[[F], F],
    /,
    caption_formatter: CaptionFormatter,
) -> CaptionedAgent | Callable[[F], F]:
    """
    Wrap an agent or agentic function in way that captures its echo stream and summarizes it.
    """
    if isinstance(magician, Agent):
        # Agent wrapper.
        return CaptionedAgent(magician, caption_formatter=caption_formatter)
    else:
        # Decorator.
        caption_logger = CaptionLogger(caption_formatter=caption_formatter)

        def wrapped_captioned(*args, **kwargs) -> Any:
            with caption_logger:
                return magician(*args, **kwargs)

        spoof_signature(wrapped_captioned, magician)
        return wrapped_captioned
