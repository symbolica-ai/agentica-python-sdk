"""
The standard logger writes short-form results to stdout,
and the whole chat history to configurable log files.

For example, to STDOUT:
```shell
Spawned Agent 25 (./logs/agent-25.log)
► Agent 25: Get one subagent to work out the 32nd power of 3, then another subagent to work out the 34th power, then return both results.
Spawned Agent 26 (./logs/agent-26.log)
► Agent 26: Work out the 32nd power of 3
◄ Agent 26: 1853020188851841
Spawned Agent 27 (./logs/agent-27.log)
► Agent 27: Work out the 34th power of 3
◄ Agent 27: 16677181699666569
◄ Agent 25: (1853020188851841, 16677181699666569)
```

This gives the user a high-level overview of what's happening in the system,
then the full chat histories for individual agents are written to files, which look roughly like this:

```file: ./logs/agent-26.log
<message role="user">
        Work out the 32nd power of 3

        When you have completed your task or query, return an instance of AgentResult[int].
</message>
<message role="agent">
        <ipython>AgentResult(result=3**32)</ipython>
</message>
```
"""

from functools import cache
from pathlib import Path
from typing import Literal, TextIO, override

from agentica.common import Chunk, Role

from ..agent_logger import AgentLogger, LogId

DEFAULT_LOGS_DIR_NAME = "logs"
DEFAULT_AGENT_FILE_PREFIX = "agent-"
DEFAULT_AGENT_FILE_SUFFIX = ".log"


# Terminal color constants
COLORS = [
    "\033[95m",  # Magenta/Pink
    "\033[94m",  # Blue
    "\033[91m",  # Red
    "\033[92m",  # Green
    "\033[93m",  # Yellow
    "\033[96m",  # Cyan
    "\033[90m",  # Dark Gray
    "\033[35m",  # Purple
    "\033[36m",  # Teal
    "\033[33m",  # Orange/Brown
    "\033[31m",  # Dark Red
]
RESET = "\033[0m"
GREY = "\033[90m"

Entity = Literal["agentic", "agent"]


def _id_to_color(id: int) -> str:
    return COLORS[id % len(COLORS)]


class StandardLogger(AgentLogger):
    local_id: LogId | None
    parent_local_id: LogId | None
    logs_dir: Path
    logs_file: Path | None
    agent_file_prefix: str
    agent_file_suffix: str
    _color: str
    _log_file_io: TextIO | None
    _short_form: bool
    _current_role: Role | None

    def __init__(
        self,
        logs_dir: Path | str | None = None,
        agent_file_prefix: str = DEFAULT_AGENT_FILE_PREFIX,
        agent_file_suffix: str = DEFAULT_AGENT_FILE_SUFFIX,
        short_form: bool = True,
    ) -> None:
        self.local_id = None
        self.logs_dir = get_logs_dir(Path(logs_dir) if logs_dir else Path(DEFAULT_LOGS_DIR_NAME))
        self.agent_file_prefix = agent_file_prefix
        self.agent_file_suffix = agent_file_suffix
        self.logs_file = None
        self._log_file_io = None
        self._color = GREY
        self._short_form = short_form
        self._current_role = None

    @override
    def on_spawn(self) -> None:
        if self.local_id is None:
            self.local_id = self._get_next_agent_id()
            self.logs_file = self._new_agent_log_file(self.local_id)
            self._color = _id_to_color(self.local_id)
        print(f"{GREY}Spawned {self._color}Agent {self.local_id} {GREY}({self.logs_file}){RESET}")

    @override
    def on_call_enter(self, user_prompt: str, parent_local_id: LogId | None = None) -> None:
        self.parent_local_id = parent_local_id
        if self.local_id is None:
            raise ValueError("on_call_enter should be called only after on_spawn")
        if self.parent_local_id is None:
            # parent_str = f"{GREY}User{RESET} "
            parent_str = ""
        else:
            parent_color = _id_to_color(int(self.parent_local_id))
            parent_str = f"{parent_color}Agent {self.parent_local_id}{RESET} "
        print(
            f"{parent_str}►  {self._color}Agent {self.local_id}{RESET}: {self._shorten(user_prompt)}"
        )

    @override
    def on_call_exit(self, result: object) -> None:
        if self.local_id is None:
            raise ValueError("on_call_exit should be called only after on_spawn")
        if self.parent_local_id is None:
            # parent_str = f"{GREY}User{RESET}"
            parent_str = ""
        else:
            parent_color = _id_to_color(int(self.parent_local_id))
            parent_str = f"{parent_color}Agent {self.parent_local_id}{RESET} "
        print(
            f"{parent_str}◄  {self._color}Agent {self.local_id}{RESET}: {self._shorten(str(result))}"
        )
        handle = self._log_file_handle()
        _ = handle.write('\n</message>\n')
        handle.flush()

    @override
    async def on_chunk(self, chunk: Chunk) -> None:
        chunk_log = ""
        if chunk.role != self._current_role:
            if self._current_role is None:
                chunk_log = f'<message role="{chunk.role}">\n\t'
            else:
                chunk_log = f'\n</message>\n<message role="{chunk.role}">\n\t'
            self._current_role = chunk.role

        if self.local_id is None:
            raise ValueError("on_chunk should be called only after on_spawn")

        chunk_log += f'{chunk.content.replace("\n", "\n\t")}'

        handle = self._log_file_handle()
        _ = handle.write(chunk_log)
        handle.flush()

    def _log_file_handle(self) -> TextIO:
        if self._log_file_io is None:
            if self.logs_file is None:
                raise ValueError("logs_file should be determined after on_spawn")
            self._log_file_io = self.logs_file.open('a')
        return self._log_file_io

    def __del__(self) -> None:
        if self._log_file_io is not None:
            self._log_file_io.close()

    def _get_next_agent_id(self) -> int:
        """
        Returns the next available agent ID by scanning existing agent log files.
        """
        logs_dir = get_logs_dir(self.logs_dir)
        existing_agents = [
            f
            for f in logs_dir.iterdir()
            if f.name.startswith(self.agent_file_prefix) and f.name.endswith(self.agent_file_suffix)
        ]
        max_agent_id = -1
        for agent_file in existing_agents:
            try:
                # Extract number between prefix and suffix
                filename = agent_file.name
                id_str = filename[len(self.agent_file_prefix) : -len(self.agent_file_suffix)]
                agent_id = int(id_str)
                max_agent_id = max(max_agent_id, agent_id)
            except ValueError:
                continue
        return max_agent_id + 1

    def _new_agent_log_file(self, agent_id: int) -> Path:
        agent_file = (
            get_logs_dir(self.logs_dir)
            / f"{self.agent_file_prefix}{agent_id}{self.agent_file_suffix}"
        )
        agent_file.touch()
        return agent_file

    def _shorten(self, msg: str) -> str:
        MAX_LEN = 80
        SUFFIX = " [...]"
        if self._short_form:
            msg = msg.strip().replace("\n", " ")
            if len(msg) > MAX_LEN:
                return msg[: MAX_LEN - len(SUFFIX)] + SUFFIX
        return msg


@cache
def get_logs_dir(logs_dir: Path) -> Path:
    """
    Returns the path to the logs directory.
    If no logs directory exists, a new one is created.
    """
    _logs_dir = logs_dir.resolve()
    if not _logs_dir.exists():
        _logs_dir.mkdir()
        # Ignore all logs so we don't clutter user working tree
        _ = (_logs_dir / ".gitignore").write_text(".gitignore\n**/*.log\n")

    return _logs_dir
