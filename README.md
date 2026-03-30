# TimeSeriesUnifiedToolbox

TimeSeriesUnifiedToolbox (TSUT) is a modular framework for building reusable time-series pipelines.
It provides typed node abstractions (data sources, transforms, models, metrics), a graph-based pipeline
definition, and component auto-registration to make experimentation and composition easier.

## Installation

### Development setup

```bash
uv sync --dev
pre-commit install
```

## Quick start

```python
from tsut.components import _auto_discovery
from tsut.core.pipeline.pipeline import Pipeline

_auto_discovery()
pipe = Pipeline()
```

## Project structure

- `src/tsut/core`: framework primitives (data types, node contracts, pipeline orchestration)
- `src/tsut/components`: concrete implementations (models, transforms, sources, metrics)
- `tests/`: unit tests
- `docs/`: project documentation and Sphinx configuration

## Documentation

This repository includes a Sphinx-compatible documentation setup in `docs/`.

- Build docs locally (after installing Sphinx):

```bash
sphinx-build -b html docs docs/_build/html
```

- Main entry page: `docs/index.rst`
- API reference page: `docs/api_reference.rst`

## Readability improvement suggestions

The following incremental improvements would further improve readability across the codebase:

- Keep docstrings consistent in style (Google or NumPy style everywhere, parsed by Sphinx Napoleon).
- Prefer one field per line for complex `Port(...)` dictionaries to improve diff readability.
- Avoid very long inline comments on configuration fields; move explanatory text to docstrings.
- Replace `print(...)` diagnostic output in library code with the project logging standard when introduced.
