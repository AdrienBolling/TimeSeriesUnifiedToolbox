"""CorrelationFilter transform node for the TSUT Framework.

Removes highly correlated features by computing a pair-wise correlation
matrix during :meth:`fit` and greedily dropping one column from every pair
whose absolute correlation exceeds a configurable threshold.

The greedy strategy iterates columns left-to-right and marks a column for
removal the first time it correlates above the threshold with a column that
has not already been marked.  This is equivalent to scikit-learn's
variance-inflation heuristic and preserves the *first* feature in every
correlated group.

* Input  – a ``(batch, feature)`` **numerical** DataFrame.
* Output – the same DataFrame with redundant features removed.

Implemented with numpy (``np.corrcoef``) for Pearson, or delegates to
``scipy.stats.spearmanr`` / ``scipy.stats.kendalltau`` for rank-based
methods.
"""

from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from tsut.components.utils.dataframe import filter_columns
from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
)
from tsut.core.nodes.node import Port
from tsut.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)

# Serialisable params: list of column names that survived the filter.
type _CorrelationFilterParams = dict[str, list[str]]


class CorrelationFilterMetadata(TransformMetadata):
    """Metadata for the CorrelationFilter node."""

    node_name: str = "CorrelationFilter"
    description: str = (
        "Remove highly correlated numerical features. "
        "Keeps only one column from every correlated pair above the threshold."
    )


class CorrelationFilterRunningConfig(TransformRunningConfig):
    """Run-time knobs that do not affect the learned parameters."""

    filtering_columns: list[str] | None = Field(
        default=None,
        description=(
            "Subset of columns to evaluate for correlation filtering. "
            "``None`` (default) evaluates all columns. "
            "Columns not in this list are always kept in the output."
        ),
    )


class CorrelationFilterHyperParameters(TransformHyperParameters):
    """Tuneable hyperparameters for the CorrelationFilter."""

    threshold: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description=(
            "Maximum allowed absolute correlation between any pair of features. "
            "When a pair exceeds this value the second column (in column order) "
            "is dropped. ``0.95`` is a common default; lower values are more "
            "aggressive."
        ),
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description=(
            "Correlation method. "
            "``'pearson'`` measures linear correlation (fast, via numpy). "
            "``'spearman'`` measures monotonic correlation (rank-based). "
            "``'kendall'`` measures ordinal association (rank-based, slower)."
        ),
    )


# Exposed at module level so external tuners can discover the search space.
hyperparameter_space: dict[str, tuple[str, Any]] = {
    "threshold": ("float", {"min": 0.5, "max": 1.0}),
    "method": ("choice", ["pearson", "spearman", "kendall"]),
}


class CorrelationFilterConfig(
    TransformConfig[
        CorrelationFilterRunningConfig,
        CorrelationFilterHyperParameters,
    ]
):
    """Full configuration for the CorrelationFilter node."""

    hyperparameters: CorrelationFilterHyperParameters = Field(
        default_factory=CorrelationFilterHyperParameters,
        description="Tuneable hyperparameters (threshold, method).",
    )
    running_config: CorrelationFilterRunningConfig = Field(
        default_factory=CorrelationFilterRunningConfig,
        description="Run-time options (filtering_columns).",
    )
    in_ports: dict[str, Port] = Field(
        default={
            "input": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch feature",
                desc="Numerical DataFrame to filter by correlation.",
            ),
        },
        description="Input ports: 'input' (numerical DataFrame).",
    )
    out_ports: dict[str, Port] = Field(
        default={
            "output": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch _",
                desc=(
                    "DataFrame with highly correlated features removed. "
                    "Feature dimension may shrink."
                ),
            ),
        },
        description="Output ports: 'output' (filtered DataFrame).",
    )


class CorrelationFilter(
    TransformNode[
        pd.DataFrame,
        TabularDataContext,
        pd.DataFrame,
        TabularDataContext,
        _CorrelationFilterParams,
    ]
):
    """Remove highly correlated numerical features.

    During :meth:`fit`, computes the pair-wise correlation matrix and
    greedily marks columns for removal when their absolute correlation
    with an already-kept column exceeds the threshold.  The first column
    in every correlated group is always retained.

    Example
    -------
    >>> cfg = CorrelationFilterConfig(
    ...     hyperparameters=CorrelationFilterHyperParameters(
    ...         threshold=0.9, method="spearman"
    ...     ),
    ... )
    >>> node = CorrelationFilter(config=cfg)
    """

    metadata = CorrelationFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: CorrelationFilterConfig) -> None:
        self._config = config
        self._params: _CorrelationFilterParams = {"columns_to_keep": []}
        self._fitted = False

    # --- TransformNode interface ------------------------------------------

    def fit(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> None:
        """Compute the correlation matrix and identify redundant columns.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.
        """
        df, _ = data["input"]
        candidates = filter_columns(
            df, self._config.running_config.filtering_columns
        )
        threshold = self._config.hyperparameters.threshold
        method = self._config.hyperparameters.method

        corr_matrix = self._compute_correlation(candidates, method)
        cols_to_drop = self._greedy_drop(candidates.columns.tolist(), corr_matrix, threshold)

        surviving_cols = [
            c for c in candidates.columns if c not in cols_to_drop
        ]

        # Columns not in the candidate set are always kept.
        non_candidate_cols = [
            c for c in df.columns if c not in candidates.columns
        ]
        self._params = {
            "columns_to_keep": non_candidate_cols + surviving_cols,
        }

    def transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Subset the DataFrame to the columns identified during fit.

        Parameters
        ----------
        data:
            Must contain key ``"input"``.
        """
        df, ctx = data["input"]
        keep = self._params["columns_to_keep"]
        dropped = [c for c in df.columns if c not in set(keep)]

        out_ctx = deepcopy(ctx)
        out_ctx.remove_columns(dropped)
        return {"output": (df[keep], out_ctx)}

    def get_params(self) -> _CorrelationFilterParams:
        """Return the list of columns that survived filtering."""
        return self._params

    def set_params(self, params: _CorrelationFilterParams) -> None:
        """Restore a previously fitted column list (checkpointing)."""
        self._params = params
        self._fitted = True

    # --- Private helpers --------------------------------------------------

    @staticmethod
    def _compute_correlation(
        df: pd.DataFrame, method: str
    ) -> np.ndarray:
        """Return an ``(n_features, n_features)`` absolute correlation matrix."""
        arr = df.to_numpy(dtype=np.float64, na_value=np.nan)

        if method == "pearson":
            # Drop rows with any NaN for corrcoef (it does not handle NaN).
            mask = ~np.isnan(arr).any(axis=1)
            clean = arr[mask]
            if clean.shape[0] < 2:
                return np.zeros((arr.shape[1], arr.shape[1]))
            return np.abs(np.corrcoef(clean, rowvar=False))

        if method == "spearman":
            from scipy.stats import spearmanr

            corr, _ = spearmanr(arr, nan_policy="omit")
            # spearmanr returns a scalar when n_features == 1.
            corr = np.atleast_2d(corr)
            return np.abs(corr)

        # method == "kendall"
        from scipy.stats import kendalltau

        n = arr.shape[1]
        corr = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mask = ~(np.isnan(arr[:, i]) | np.isnan(arr[:, j]))
                if mask.sum() < 2:
                    corr[i, j] = corr[j, i] = 0.0
                else:
                    tau, _ = kendalltau(arr[mask, i], arr[mask, j])
                    corr[i, j] = corr[j, i] = abs(tau)
        return corr

    @staticmethod
    def _greedy_drop(
        columns: list[str],
        corr_matrix: np.ndarray,
        threshold: float,
    ) -> set[str]:
        """Greedily select columns to drop from a correlation matrix.

        Iterates in column order.  For every pair ``(i, j)`` with ``i < j``
        where ``|corr| > threshold``, column *j* is marked for removal
        (provided column *i* has not already been marked).
        """
        n = len(columns)
        to_drop: set[int] = set()

        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, n):
                if j in to_drop:
                    continue
                if corr_matrix[i, j] > threshold:
                    to_drop.add(j)

        return {columns[idx] for idx in to_drop}
