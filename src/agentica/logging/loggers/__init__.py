"""
A series of standard loggers.
"""

from .file_logger import FileLogger
from .print_logger import PrintLogger
from .standard_logger import StandardLogger
from .stream_logger import StreamLogger

__all__ = ['FileLogger', 'PrintLogger', 'StandardLogger', 'StreamLogger']
