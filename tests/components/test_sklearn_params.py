"""Tests for :mod:`tsut.components.utils.sklearn_params`."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from tsut.components.utils.sklearn_params import (
    get_sklearn_fitted_params,
    set_sklearn_fitted_params,
)


def _trained_estimator() -> LinearRegression:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 2))
    y = X @ np.array([1.0, 2.0])
    est = LinearRegression()
    est.fit(X, y)
    return est


class TestFittedParamsRoundtrip:
    def test_get_returns_only_fitted_attributes(self) -> None:
        est = _trained_estimator()
        fitted = get_sklearn_fitted_params(est)

        # sklearn's trailing-underscore convention.
        assert "coef_" in fitted
        assert "intercept_" in fitted
        # Dunders and non-underscore attrs must be filtered out.
        for name in fitted:
            assert not name.startswith("_")
            assert name.endswith("_")
            assert not name.endswith("__")

    def test_set_restores_behaviour_on_fresh_estimator(self) -> None:
        est = _trained_estimator()
        fitted = get_sklearn_fitted_params(est)

        reborn = LinearRegression()
        set_sklearn_fitted_params(reborn, fitted)

        rng = np.random.default_rng(1)
        X_test = rng.standard_normal((5, 2))
        np.testing.assert_allclose(est.predict(X_test), reborn.predict(X_test))
