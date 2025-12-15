from agentica_internal.internal_errors import *

__all__ = [
    # base exception
    'AgenticaError',
    # connection errors
    'ConnectionError',
    'WebSocketConnectionError',
    'WebSocketTimeoutError',
    # server/generation errors
    'ServerError',
    'APIConnectionError',
    'APITimeoutError',
    'BadRequestError',
    'ConflictError',
    'ContentFilteringError',
    'DeadlineExceededError',
    'GenerationError',
    'InferenceError',
    'InternalServerError',
    'UsageError',
    'MaxTokensError',
    'MaxRoundsError',
    'NotFoundError',
    'OverloadedError',
    'PermissionDeniedError',
    'InsufficientCreditsError',
    'RateLimitError',
    'RequestTooLargeError',
    'ServiceUnavailableError',
    'UnauthorizedError',
    'UnprocessableEntityError',
    # invocation errors
    'InvocationError',
    'TooManyInvocationsError',
    'NotRunningError',
    # bugs
    'ThisIsABug',
]


class InvalidAPIKey(AgenticaError):
    """Local invalid API key exception."""
