# AGENTS.md

## Purpose
This repository contains a Python application with a `src/` layout and data artifacts.
These instructions are intended to help contributors and agents operate safely and
consistently, especially when running in constrained environments.

## Repository layout
- `src/`: application code
- `data/`: input datasets or assets
- `out/`: generated artifacts/results
- `pyproject.toml`: project configuration
- `README.md`: usage and high-level documentation

## General guidance
- Prefer static inspection over executing code whenever possible.
- When running commands, keep them fast and scoped (avoid recursive `ls` on large trees).
- If needed, use `uv` to manage project environment. See `README.md` for usage example.

## Code style & conventions
- Respect existing formatting and naming conventions in the codebase.
- For internal utility functions, avoid excessive argument validation when the properties
  can be guaranteed by the caller. Document the constraints on the arguments.
- Favor small, focused changes; avoid sweeping refactors unless explicitly requested.
- Be exhaustive in considering edge cases.
- Maintain docstrings and type annotations where present.
- Use modern typing, i.e. `list`, `tuple` instead of `List`, `Tuple` or importing
  `annotations` from `__future__`.

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
