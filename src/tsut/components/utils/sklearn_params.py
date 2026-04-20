"""Helpers for extracting and restoring the fitted state of sklearn estimators.

scikit-learn stores trained weights/learned state on attributes whose names end
with a trailing underscore (e.g. ``coef_``, ``classes_``, ``estimators_``) —
this is a strict convention across the library. ``BaseEstimator.get_params`` /
``set_params`` only cover **constructor** hyperparameters, not this fitted
state, so these helpers fill the gap.
"""

from typing import Any

from sklearn.base import BaseEstimator


def get_sklearn_fitted_params(estimator: BaseEstimator) -> dict[str, Any]:
    """Return the fitted attributes of a sklearn estimator.

    Fitted attributes follow the sklearn convention of ending with a single
    trailing underscore. Dunder attributes and private attributes are excluded.
    """
    return {
        name: value
        for name, value in vars(estimator).items()
        if name.endswith("_") and not name.startswith("_") and not name.endswith("__")
    }


def set_sklearn_fitted_params(
    estimator: BaseEstimator, fitted_params: dict[str, Any]
) -> BaseEstimator:
    """Restore fitted attributes onto a sklearn estimator in-place."""
    for name, value in fitted_params.items():
        setattr(estimator, name, value)
    return estimator
