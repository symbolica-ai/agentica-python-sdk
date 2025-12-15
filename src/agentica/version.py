from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

try:
    __version__ = get_version("symbolica-agentica")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
