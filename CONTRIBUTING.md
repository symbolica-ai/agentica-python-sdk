# Contributing to Agentica Python SDK

Thank you for your interest in contributing to Agentica! We welcome contributions from the community.

## Reporting Issues

Please report bugs and feature requests at [agentica-issues](https://github.com/symbolica-ai/agentica-issues).

## Development Setup

### Prerequisites

- Python 3.12 or 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```sh
git clone https://github.com/symbolica-ai/agentica-python-sdk.git
cd agentica-python-sdk
uv sync
```

> **Note:** Local development requires a running Agentica server. See [agentica-server](https://github.com/symbolica-ai/agentica-server) for setup instructions.

### Running Tests

Ensure that you have `uv` installed, then

```bash
uv run pytest
```

> **Note:** This test suite includes unit tests only. Additional integration and system tests are run internally.

### Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```sh
uv run ruff check --fix .
uv run ruff format .
```

## Contributor License Agreement

By contributing to this project, you agree to the terms outlined in our [Contributor License Agreement (CLA)](./CLA.md). 

When you submit your first pull request, a CLA bot will comment with instructions to sign electronically. This is a one-time process that takes just a minute.

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Open a pull request

Keep changes focused, include tests for new functionality, and follow existing code style.

## Code of Conduct

See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

