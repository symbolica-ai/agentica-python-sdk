import os
import sys

from agentica.common import ModelStrings

__all__ = ['print_model_notice']


_model_notices_printed: set[str] = set()
_MODEL_NOTICES: dict[ModelStrings, str] = {
    "openai:gpt-4o": "has known task performance issues, consider using a more capable model.",
    "openai:gpt-5": "is subject to high latency and low throughput, consider using another model if performance is critical.",
}


def print_model_notice(model: ModelStrings) -> None:
    if os.environ.get('AGENTICA_NO_MODEL_NOTICE'):
        return

    if model in _model_notices_printed:
        return

    _model_notices_printed.add(model)

    if model in _MODEL_NOTICES:
        suffix = "(set AGENTICA_NO_MODEL_NOTICE to disable this message)"
        print(f"{model} {_MODEL_NOTICES[model]} {suffix}", file=sys.stderr)
