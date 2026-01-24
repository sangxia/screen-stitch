# AGENTS.md

## Purpose
This repository appears to be a Python application with a `src/` layout and data artifacts. These instructions are intended to help contributors and agents operate safely and consistently, especially when running in constrained environments.

## Repository layout
- `src/`: application code
- `data/`: input datasets or assets
- `out/`: generated artifacts/results
- `pyproject.toml`: project configuration
- `README.md`: usage and high-level documentation

## General guidance
- Prefer static inspection over executing code whenever possible.
- When running commands, keep them fast and scoped (avoid recursive `ls` on large trees).
- Use `rg` (ripgrep) for search instead of `grep -R`.

## Code style & conventions
- Respect existing formatting and naming conventions in the codebase.
- Favor small, focused changes; avoid sweeping refactors unless explicitly requested.
- Maintain docstrings and type annotations where present.

## Testing & validation
- Do not run tests unless explicitly requested.
- If tests are requested, prefer targeted test runs over full suites.
- Document any limitations (e.g., missing dependencies, environment constraints).

## Data handling
- Treat `data/` and `out/` as non-code artifacts.
- Do not modify or regenerate data unless explicitly instructed.

## Communication standards
- Summarize changes clearly.
- When suggesting fixes, provide file paths and identifiers to locate the relevant code.
