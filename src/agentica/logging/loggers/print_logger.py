"""
Like StandardLogger, but only writes short-form results to stdout.
"""

from pathlib import Path
from typing import override

from agentica.common import Chunk

from ..agent_logger import LogId
from .standard_logger import GREY, RESET, StandardLogger, _id_to_color


class PrintLogger(StandardLogger):
    local_id: LogId | None
    parent_local_id: LogId | None
    logs_file: Path | None
    _color: str
    _id_counter: int

    def __init__(self) -> None:
        super().__init__()
        self._id_counter = 0

    @override
    def on_spawn(self) -> None:
        if self.local_id is not None:
            raise ValueError("on_spawn should be called only once")
        self.local_id = self._get_next_agent_id()
        self.logs_file = self._new_agent_log_file(self.local_id)
        self._color = _id_to_color(self.local_id)
        print(f"{GREY}Spawned {self._color}Agent {self.local_id}{RESET}")

    @override
    async def on_chunk(self, chunk: Chunk) -> None:
        pass

    @override
    def _new_agent_log_file(self, agent_id: int) -> Path:
        return Path("/dev/null")

    @override
    def _get_next_agent_id(self) -> int:
        self._id_counter += 1
        return self._id_counter
