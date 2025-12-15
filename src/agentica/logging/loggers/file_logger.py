"""
Like StandardLogger, but only writes chat history to file.
"""

from pathlib import Path
from typing import override

from ..agent_logger import LogId
from .standard_logger import StandardLogger


class FileLogger(StandardLogger):
    local_id: LogId | None
    parent_local_id: LogId | None
    logs_file: Path | None

    @override
    def on_spawn(self) -> None:
        if self.local_id is not None:
            raise ValueError("on_spawn should be called only once")
        self.local_id = self._get_next_agent_id()
        self.logs_file = self._new_agent_log_file(self.local_id)

    @override
    def on_call_enter(self, user_prompt: str, parent_local_id: LogId | None = None) -> None:
        self.parent_local_id = parent_local_id

    @override
    def on_call_exit(self, result: object) -> None:
        handle = self._log_file_handle()
        _ = handle.write('\n</message>\n')
        handle.flush()
