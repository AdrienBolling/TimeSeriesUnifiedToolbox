"""Builders for ``TabularData`` / ``TabularDataContext`` test fixtures.

Centralising the construction of tabular test data keeps every test file
working from the same, deterministic inputs. All helpers return
``(DataFrame, TabularDataContext)`` tuples in the exact shape the TSUT
runners expect on a node port.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsut.core.common.data.data import (
    CategoricalData,
    MixedData,
    NumericalData,
    TabularDataContext,
)


def numerical_df(
    n_rows: int = 20,
    n_cols: int = 3,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a deterministic numerical DataFrame.

    Args:
        n_rows: Number of rows in the dataset.
        n_cols: Number of float columns.
        seed: Seed for the internal RNG to keep tests reproducible.

    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(size=(n_rows, n_cols)).astype(np.float64)
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(data, columns=cols)


def numerical_context(df: pd.DataFrame) -> TabularDataContext:
    """Build a ``TabularDataContext`` that marks every column as numerical."""
    return TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=[NumericalData for _ in df.columns],
    )


def numerical_pair(
    n_rows: int = 20,
    n_cols: int = 3,
    *,
    seed: int = 0,
) -> tuple[pd.DataFrame, TabularDataContext]:
    """Return a ``(DataFrame, NumericalContext)`` tuple ready for a port."""
    df = numerical_df(n_rows, n_cols, seed=seed)
    return df, numerical_context(df)


def linear_regression_dataset(
    n_rows: int = 40,
    n_features: int = 3,
    *,
    seed: int = 0,
    noise_scale: float = 0.01,
) -> tuple[
    tuple[pd.DataFrame, TabularDataContext],
    tuple[pd.DataFrame, TabularDataContext],
    np.ndarray,
]:
    """Create a synthetic linear-regression dataset.

    The features are i.i.d. standard normal, the targets are a fixed
    linear combination of the features plus a tiny bit of Gaussian
    noise so a ``LinearRegression`` should achieve near-zero MSE.

    Returns:
        ``((X_df, X_ctx), (y_df, y_ctx), true_coefs)`` where ``true_coefs``
        is the vector of ground-truth coefficients used to synthesise y.

    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_rows, n_features)).astype(np.float64)
    coefs = np.linspace(1.0, 2.0, n_features, dtype=np.float64)
    noise = rng.standard_normal(size=n_rows) * noise_scale
    y = X @ coefs + noise
    y = y.reshape(-1, 1)

    x_cols = [f"x{i}" for i in range(n_features)]
    y_cols = ["target"]
    X_df = pd.DataFrame(X, columns=x_cols)
    y_df = pd.DataFrame(y, columns=y_cols)

    return (X_df, numerical_context(X_df)), (y_df, numerical_context(y_df)), coefs


def mixed_df() -> pd.DataFrame:
    """Return a small DataFrame with one numerical and one categorical column."""
    return pd.DataFrame(
        {
            "num": pd.Series([1.0, 2.0, 3.0, 4.0], dtype="float64"),
            "cat": pd.Series(["a", "b", "a", "c"], dtype="object"),
        }
    )


def mixed_context(df: pd.DataFrame) -> TabularDataContext:
    """Build a mixed numerical/categorical context from *df*."""
    cats: list[type] = []
    for dtype in df.dtypes:
        if pd.api.types.is_numeric_dtype(dtype):
            cats.append(NumericalData)
        else:
            cats.append(CategoricalData)
    return TabularDataContext(
        columns=list(df.columns),
        dtypes=list(df.dtypes),
        categories=cats,
    )


def scalar_numerical_context(col_name: str = "score") -> TabularDataContext:
    """Single-cell numerical context matching the MSE/MAE metric outputs."""
    return TabularDataContext(
        columns=[col_name],
        dtypes=[np.dtype("float64")],
        categories=[NumericalData],
    )


def as_mixed_categories(df: pd.DataFrame) -> list[type]:
    """Return a list of ``MixedData`` categories matching ``df.columns``."""
    return [MixedData for _ in df.columns]
