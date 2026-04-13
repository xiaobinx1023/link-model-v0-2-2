# AGENTS.md

## Project goal
Build a Python model from technical papers and user collateral such as codes, equations, data.

## Global rules
- Never invent equations that are not grounded in the source materials or explicit user assumptions.
- Separate source-derived assumptions from user-supplied assumptions.
- Keep modeling code modular and testable.
- keep single responsiblity principle.
- Every test must map to one item in docs/spec_sheet.yaml.
- Create the interactive plot interface to inspect the model results.
- Save plots to artifacts/figures/.
- Save machine-readable result tables to artifacts/tables/.
- Save presentation output to artifacts/slides/.

## Coding standards
<!-- - Python 3.11+ -->
- Use dataclasses or pydantic for config objects
- Use pytest for tests
- Prefer numpy/scipy/pandas/matplotlib
- Avoid hidden global state
- Add docstrings with equation references where relevant

## Output expectations
