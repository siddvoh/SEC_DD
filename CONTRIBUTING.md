# Contributing to SEC Deep Dive

Thanks for your interest in contributing! Here's how to get started.

## Setup

1. Fork and clone the repo
2. Install in editable mode with test extras: `pip install -e ".[test]"`
3. Copy `.env.example` to `.env` and add your OpenAI API key
4. Run the tests: `pytest -q`
5. Test the CLI: `secdd "What are the main risk factors in retail?"`

## What to Work On

- Bug fixes and improvements to filing parsing (`secdd/edgar.py`)
- Better prompts for specific filing types (`secdd/prompts.py`)
- CLI improvements (`secdd/cli.py`)
- Support for additional document types
- Performance optimizations
- New EDGAR form types and data sources

Check the [Issues](https://github.com/siddvoh/secdd/issues) tab for open tasks.

## Pull Requests

1. Create a branch from `main`
2. Make your changes
3. Test locally (both CLI and library usage if applicable)
4. Submit a PR with a clear description of what changed and why

Keep PRs focused on a single change. If you're fixing a bug and also want to refactor nearby code, split them into separate PRs.

## Code Style

- Python: follow existing patterns in the codebase
- Use conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- No new dependencies without discussion in an issue first

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your Python version and OS
