"""LinearRegression model node for the TSUT Framework.

Wraps ``sklearn.linear_model.LinearRegression`` and exposes it as a TSUT
:class:`~tsut.core.nodes.models.model.Model` node.  The node expects two
input ports:

* ``X`` – feature matrix ``(batch, features)`` as a numpy array
* ``y`` – target matrix ``(batch, targets)`` as a numpy array (training / evaluation only)

and emits one output port:

* ``pred`` – predicted values ``(batch, targets)`` as a numpy array

sklearn handles multi-output regression transparently when ``y`` is 2-D, so
no extra wrapping is needed.
"""

from typing import Any

import numpy as np
from pydantic import Field
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from tsut.components.utils.sklearn_params import (
    get_sklearn_fitted_params,
    set_sklearn_fitted_params,
)
from tsut.core.common.enums import NodeExecutionMode
from tsut.core.nodes.models.model import (
    Model,
    ModelConfig,
    ModelHyperParameters,
    ModelMetadata,
    ModelRunningConfig,
)
from tsut.core.nodes.node import Port


class LinearRegressionMetadata(ModelMetadata):
    node_name: str = "LinearRegression"
    description: str = (
        "Linear Regression based on scikit-learn. "
        "Supports both single- and multi-output regression."
    )


class LinearRegressionHyperParameters(ModelHyperParameters):
    pass


class LinearRegressionRunningConfig(ModelRunningConfig):
    fit_intercept: bool = Field(default=True, description="Whether to calculate the intercept for this model.")


hyperparameter_space: dict[str, Any] = {}

type _LRParams = dict[str, Any]


class LinearRegressionConfig(
    ModelConfig[LinearRegressionHyperParameters, LinearRegressionRunningConfig]
):
    hyperparameters: LinearRegressionHyperParameters = Field(default_factory=LinearRegressionHyperParameters)
    running_config: LinearRegressionRunningConfig = Field(default_factory=LinearRegressionRunningConfig)
    in_ports: dict[str, Port] = Field(
        default={
            "X": Port(arr_type=ArrayLikeEnum.NUMPY, data_structure=DataStructureEnum.TABULAR, data_category=DataCategoryEnum.MIXED, data_shape="batch features", desc="Input feature matrix."),
            "y": Port(arr_type=ArrayLikeEnum.NUMPY, data_structure=DataStructureEnum.TABULAR, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch targets", mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION], desc="Target values."),
        },
    )
    out_ports: dict[str, Port] = Field(
        default={"pred": Port(arr_type=ArrayLikeEnum.NUMPY, data_structure=DataStructureEnum.TABULAR, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch targets", desc="Predicted values.")},
    )


class LinearRegressionNode(Model[np.ndarray, TabularDataContext, np.ndarray, TabularDataContext, _LRParams]):
    metadata = LinearRegressionMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: LinearRegressionConfig) -> None:
        self._config = config
        self._model = SklearnLinearRegression(
            fit_intercept=config.running_config.fit_intercept,
        )
        self._target_context_dump: dict[str, list[str]] = {}

    def fit(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        X, _ = data["X"]
        y, y_ctx = data["y"]
        self._model.fit(X, y)
        self._target_context_dump = y_ctx.dump_dict

    def predict(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        X, _ = data["X"]
        pred: np.ndarray = self._model.predict(X)
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        pred_ctx = tabular_context_from_dict_dump(self._target_context_dump)
        return {"pred": (pred, pred_ctx)}

    def get_params(self) -> _LRParams:
        return {
            "fitted_params": get_sklearn_fitted_params(self._model),
            "target_context": self._target_context_dump,
        }

    def set_params(self, params: _LRParams) -> None:
        set_sklearn_fitted_params(self._model, params["fitted_params"])
        self._target_context_dump = params["target_context"]
