import inspect
from inspect import iscoroutinefunction
from typing import Callable


def wrap_callable[**I, O](can_be_called: Callable[I, O]) -> Callable[I, O]:
    if iscoroutinefunction(can_be_called):

        async def wrapped_coro(*args: I.args, **kwargs: I.kwargs) -> O:
            return await can_be_called(*args, **kwargs)

        wrapped = wrapped_coro
    else:

        def wrapped_sync(*args: I.args, **kwargs: I.kwargs) -> O:
            return can_be_called(*args, **kwargs)

        wrapped = wrapped_sync

    spoof_signature(wrapped, can_be_called)
    return wrapped


def spoof_signature[**I, O](this: Callable[I, O], that: Callable[I, O]) -> None:
    for attr in [
        "__name__",
        "__doc__",
        "__qualname__",
        "__module__",
        "__annotations__",
        "__defaults__",
        "__kwdefaults__",
    ]:
        if hasattr(that, attr):
            setattr(this, attr, getattr(that, attr))

    if hasattr(that, "__signature__"):
        setattr(this, "__signature__", getattr(that, "__signature__"))
    else:
        setattr(this, "__signature__", inspect.signature(that))
    if hasattr(that, "__text_signature__"):
        setattr(this, "__text_signature__", getattr(that, "__text_signature__"))
