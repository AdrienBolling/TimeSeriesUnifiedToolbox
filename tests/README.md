# TSUT Test Suite

## Layout

```
tests/
├── conftest.py              Shared pytest fixtures (numerical df / context,
│                            regression dataset, CSV + JSON tmp fixture, rng).
├── shims/                   Reusable test doubles & builders
│   ├── tabular.py             Deterministic DataFrame / Context builders
│   ├── nodes.py               ConstantSource, IdentityTransform, MeanModel,
│   │                          SumCountMetric — lightweight Node test doubles
│   └── pipelines.py           `build_source_model_sink_pipeline` factory
├── core/                    Tests for the framework itself
│   ├── common/                data.py, registry.py, version.py, typeguards.py
│   ├── nodes/                 Node base, registry, Sink, Transform, Model, Metric
│   └── pipeline/              Pipeline, SmartRunner, PipelineRunnerWrapper
└── components/              Tests for built-in component nodes
    ├── test_inputs_passthrough.py
    ├── test_tabular_csv_fetcher.py
    ├── test_standard_scaler.py
    ├── test_linear_regression.py
    ├── test_mse.py
    └── test_sklearn_params.py
```

## Conventions

* **Determinism** — every dataset is built with an explicit seed (see
  `tests/shims/tabular.py`). No tests depend on wall-clock or unseeded RNGs.
* **Fixtures live in `conftest.py`** — never import `pytest` fixtures directly
  from other test modules. Builders that do not need the pytest machinery
  live in `tests/shims/` and can be imported as plain functions.
* **Shims over mocks** — component contracts are easy enough to satisfy with
  real (but minimal) Node subclasses, which keeps tests honest about the
  `Node` / `DataSourceNode` / `Model` / `MetricNode` metaclass / lifecycle
  plumbing.
* **Tests stay lean** — one behaviour per test function, no hidden state,
  AAA ordering (arrange / act / assert).
* **Auto-registration** — importing `tsut` in `conftest.py` triggers the
  component auto-discovery. Tests that look up nodes by name in the global
  `NODE_REGISTRY` can rely on that import having already happened.

## Running

```bash
uv run pytest                       # full suite
uv run pytest tests/core            # core only
uv run pytest -k linear_regression  # name filter
uv run pytest --cov=tsut            # with coverage
```

## Adding tests

* Put reusable fixtures in `conftest.py` if they are needed by multiple
  modules. Keep single-use helpers local to the test file.
* Put reusable Node subclasses or dataset builders under `tests/shims/`;
  they should be deterministic and importable outside pytest.
* Mirror the package layout: a test for
  `src/tsut/core/common/foo.py` lives in `tests/core/common/test_foo.py`.
