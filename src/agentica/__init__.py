from sys import version_info

import agentica.flags as flags

from .agent import Agent, spawn
from .coming_soon import ComingSoon
from .common import ModelStrings, last_usage, total_usage
from .decorator import agentic
from .function import AgenticFunction
from .template import template
from .version import __version__

if not flags.in_testing:
    from agentica_internal.telemetry import initialize_tracing

    from agentica.sdk_logging import enable_sdk_logging

    # Initialize OpenTelemetry tracing for distributed tracing when SDK is imported
    initialize_tracing(
        service_name="agentica-sdk-python",
        instrument_httpx=True,
        log_level='DEBUG',
    )

    # turn on tagalog logging to a ring buffer, to create a local log file when
    # errors occur
    enable_sdk_logging()


if not (3, 12) <= version_info[:2] <= (3, 14):
    raise RuntimeError("Agentica only supports Python versions 3.12.x - 3.14.x in this release.")

__all__ = [
    "agentic",
    "spawn",
    "Agent",
    "ModelStrings",
    "ComingSoon",
    "last_usage",
    "total_usage",
    "template",
    "__version__",
]


from agentica_internal.warpc import forbidden

forbidden.whitelist_modules("agentica.std.")
forbidden.whitelist_objects(
    spawn,
    Agent.call,
    Agent.spawn,
    AgenticFunction,
    AgenticFunction.__call__,
    agentic,
)
