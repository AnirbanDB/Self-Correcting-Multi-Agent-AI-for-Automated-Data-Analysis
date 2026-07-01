# Automated Data Analysis with LLM Server

## Features

- **Master Agent**: Decomposes complex data analysis requests into manageable sub-problems.
- **Multi-Agent Architecture**: Solves sub-problems by generating and executing Python code.
- **Analysis Agent**: Specialized agent for handpick and analyzing plotted diagrams and visual outputs.
- **Code Agent**: Executes Python code and captures output or error messages.
- **Session Workspace**: Persists generated graphs, code, and diagrams to enable contextual multi-turn conversations.
- **Centralized Logging**: Unified logger for tracking user requests and system activity.
- **Session Logging**: Detailed logs for easier debugging.

## Getting Started

### Prerequisites

1. Ensure you have a `.env` file with the following variables:

- `OPENAI_API_KEY`

2. Ensure poetry is installed on your device

### Running the Server

```shell
poetry install && poetry run python3 main.py --env local|dev|prod --debug
```

## Roadmap

- Sandbox environment for running Python code safely.
- Support for complex graph structures instead of linear execution graphs.
