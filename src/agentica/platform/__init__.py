from pathlib import Path

from platformdirs import user_cache_dir

__all__ = ['AGENTICA_USER_DIR', 'AGENTICA_ERROR_LOG_DIR']

AGENTICA_USER_DIR = Path(user_cache_dir('Agentica'))
AGENTICA_ERROR_LOG_DIR = AGENTICA_USER_DIR / 'error_logs'
