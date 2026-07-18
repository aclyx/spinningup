# Repository Agent Guidance

## Python dependency management

- Prefer `uv` for Python environments and dependency installation in this repository.
- Create the local environment with `uv venv`.
- Install the project and its dependencies with `uv pip install -e .`.
- Avoid invoking `pip` directly unless the user explicitly requests it or `uv` cannot support the required operation.
