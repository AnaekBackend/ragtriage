# Contributing to RAGTriage

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ragtriage`
3. Install dependencies: `uv sync`
4. Create a branch: `git checkout -b feature/my-feature`

## Development

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black src/
uv run ruff check src/
```

## Pull Request Process

1. Ensure tests pass
2. Update documentation if needed
3. Create PR with clear description of changes
4. Link any related issues

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to public functions
- Keep functions focused and small

## Questions?

Open an issue for discussion before large changes.
