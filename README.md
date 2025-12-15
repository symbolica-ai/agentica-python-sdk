# Agentica Python SDK

[![PyPI version](https://img.shields.io/pypi/v/symbolica-agentica.svg)](https://pypi.org/project/symbolica-agentica/)

[Agentica](https://agentica.symbolica.ai) is a type-safe AI framework that lets LLM agents integrate with your code—functions, classes, live objects, even entire SDKs. Instead of building MCP wrappers or brittle schemas, you pass references directly; the framework enforces your types at runtime, constrains return types, and manages agent lifecycle.

## Documentation

The full documentation can be found at [docs.symbolica.ai](https://docs.symbolica.ai).

## Installation

```sh
pip install symbolica-agentica
```

Grab an API key [here](https://www.symbolica.ai/agentica).

```sh
export AGENTICA_API_KEY=<your-api-key>
```

**Want to run locally?** Run the [Agentica Server](https://github.com/symbolica-ai/agentica-server).

## Quick Example

```python
from agentica import agentic
from typing import Literal

@agentic()
async def analyze(text: str) -> Literal["positive", "neutral", "negative"]:
    """Analyze sentiment"""
    ...

result = await analyze("Agentica is an amazing framework!")
```

See the [Quickstart Guide](https://docs.symbolica.ai/quickstart) for a complete walkthrough.

## Requirements

Python 3.12 or 3.13, `uv`.

## Issues

Please report bugs, feature requests, and other issues in the [symbolica/agentica-issues](https://github.com/symbolica-ai/agentica-issues) repository.

## Contributing

See [CONTRIBUTING.md](https://github.com/symbolica-ai/agentica-python-sdk/blob/main/CONTRIBUTING.md) for guidelines. All contributors must agree to our [CLA](https://github.com/symbolica-ai/agentica-python-sdk/blob/main/CLA.md).

## Code of Conduct

This project adheres to a [Code of Conduct](https://github.com/symbolica-ai/agentica-python-sdk/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

See [LICENSE](https://github.com/symbolica-ai/agentica-python-sdk/blob/main/LICENSE).

