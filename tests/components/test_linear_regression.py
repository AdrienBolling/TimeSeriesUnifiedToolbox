"""Tests for :class:`tsut.components.nodes.models.linear_regression.LinearRegressionNode`."""

from __future__ import annotations

import numpy as np

from tsut.components.nodes.models.linear_regression import (
    LinearRegressionConfig,
    LinearRegressionNode,
)


class TestLinearRegression:
    def test_fit_recovers_true_coefficients(self, regression_dataset) -> None:
        (X_pair, y_pair, coefs) = regression_dataset
        X_df, X_ctx = X_pair
        y_df, y_ctx = y_pair

        node = LinearRegressionNode(config=LinearRegressionConfig())
        node.fit(
            {
                "X": (X_df.to_numpy(), X_ctx),
                "y": (y_df.to_numpy(), y_ctx),
            }
        )
        fitted_coefs = node.get_params()["fitted_params"]["coef_"].flatten()
        np.testing.assert_allclose(fitted_coefs, coefs, atol=0.05)

    def test_predict_returns_2d_array(self, regression_dataset) -> None:
        (X_pair, y_pair, _) = regression_dataset
        X_df, X_ctx = X_pair
        y_df, y_ctx = y_pair

        node = LinearRegressionNode(config=LinearRegressionConfig())
        node.fit(
            {
                "X": (X_df.to_numpy(), X_ctx),
                "y": (y_df.to_numpy(), y_ctx),
            }
        )
        out = node.predict({"X": (X_df.to_numpy(), X_ctx)})
        pred, pred_ctx = out["pred"]
        assert pred.ndim == 2
        assert pred.shape == (X_df.shape[0], 1)
        assert pred_ctx.columns == y_ctx.columns

    def test_params_round_trip(self, regression_dataset) -> None:
        (X_pair, y_pair, _) = regression_dataset
        X_df, X_ctx = X_pair
        y_df, y_ctx = y_pair

        node = LinearRegressionNode(config=LinearRegressionConfig())
        node.fit(
            {
                "X": (X_df.to_numpy(), X_ctx),
                "y": (y_df.to_numpy(), y_ctx),
            }
        )
        params = node.get_params()

        reborn = LinearRegressionNode(config=LinearRegressionConfig())
        reborn.set_params(params)
        pred_before, _ = node.predict({"X": (X_df.to_numpy(), X_ctx)})["pred"]
        pred_after, _ = reborn.predict({"X": (X_df.to_numpy(), X_ctx)})["pred"]
        np.testing.assert_array_equal(pred_before, pred_after)
