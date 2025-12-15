from collections.abc import Callable
from contextlib import contextmanager

from agentica_internal.core.log import (
    clear_ring_buffer,
    get_log_tags,
    set_log_tags,
    unset_log_streams,
)

__all__ = ['enable_sdk_logging', 'with_sdk_logging']


def enable_sdk_logging(*, log_tags: str = '1') -> Callable[[], None]:
    reset_streams = unset_log_streams()
    reset_tags = set_log_tags(log_tags or get_log_tags())
    clear_ring_buffer()

    def reset_state():
        reset_streams()
        reset_tags()

    return reset_state


@contextmanager
def with_sdk_logging(*, log_tags: str = '1'):
    reset = enable_sdk_logging(log_tags=log_tags)
    yield
    reset()
