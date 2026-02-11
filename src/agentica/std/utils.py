import inspect
from typing import Callable


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
