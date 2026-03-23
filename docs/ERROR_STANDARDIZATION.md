# Error standardization suggestions

This library currently raises a mix of built-in exceptions (`ValueError`, `TypeError`, `FileNotFoundError`, `NotImplementedError`).
To make error handling consistent for users, you can standardize around a small exception hierarchy.

## Suggested hierarchy

```python
class TsutError(Exception):
    """Base exception for the library."""


class TsutConfigurationError(TsutError):
    """Invalid configuration for nodes/pipelines."""


class TsutDataValidationError(TsutError):
    """Input data failed schema/type/value checks."""


class TsutPipelineError(TsutError):
    """Pipeline graph, compilation, or execution invariants failed."""


class TsutRegistryError(TsutError):
    """Node registration or lookup failed."""


class TsutDataSourceError(TsutError):
    """Loading/parsing from external sources failed."""
```

## Practical mapping from current code

- Use **`TsutConfigurationError`** where node/pipeline options are invalid.
- Use **`TsutDataValidationError`** for shape/type/content constraints on input data.
- Use **`TsutPipelineError`** for DAG validity/execution-order or runtime pipeline invariants.
- Use **`TsutRegistryError`** for duplicate registration and missing-node lookups.
- Use **`TsutDataSourceError`** for CSV/JSON fetch or parse failures.

## Why this is useful

- Users can catch one type (`TsutError`) for all library-originated failures.
- Users can still catch specific subtypes when they need targeted recovery behavior.
- Error semantics become clearer than broad `ValueError` usage.
- Future API changes stay easier to manage with a stable exception contract.

## Minimal adoption path

1. Add the hierarchy in a single module, e.g. `src/tsut/core/common/errors.py`.
2. Re-export from package `__init__.py` so users import from one stable location.
3. Convert raises incrementally by subsystem:
   - pipeline and runners
   - registry
   - data sources
   - transforms
4. Keep existing error messages unchanged where possible to avoid breaking tests/docs.

