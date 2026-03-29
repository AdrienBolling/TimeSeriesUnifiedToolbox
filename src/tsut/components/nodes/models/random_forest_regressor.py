"""RandomForestRegressor node implementation module."""


from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

from tsut.core.common.data.data import ArrayLikeEnum, DataCategoryEnum
from tsut.core.common.data.tabular_data import (
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from tsut.core.nodes.models.model import (
    Model,
    ModelConfig,
    ModelHyperParameters,
    ModelMetadata,
    ModelRunningConfig,
)
from tsut.core.nodes.node import Port


class RandomForestMetadata(ModelMetadata):
    """Metadata for the RandomForestRegressor model."""

    name: str = "RandomForestRegressor"
    description: str = "Random Forest Regressor model based on scikit-learn implementation."

class RandomForestRegressorRunningConfig(ModelRunningConfig):
    """Running configuration for the RandomForestRegressor model."""

    criterion: str = "squared_error"  # The function to measure the quality of a split. Supported criteria are “squared_error” for mean squared error, which is equal to variance reduction as feature selection criterion, and “absolute_error” for mean absolute error, which is equal to mean absolute deviation reduction as feature selection criterion.
    random_state: int | None = None  # Controls the randomness of the estimator. The features are always randomly permuted at each split. When max_features < n_features, the algorithm will select max_features at random for each split before finding the best split among them. So, the best found split may vary, even with the same training data, if the improvement of the criterion is identical for several splits and if the improvement is smaller than the machine precision. When random_state is not None, random_state is used as the seed of the pseudo random number generator to select a random feature to split on at each node when max_features < n_features. This ensures that the randomness of each tree is independent across different calls to fit, and that the results are reproducible across different calls to fit.

class RandomForestRegressorHyperParameters(ModelHyperParameters):
    """Hyperparameters for the RandomForestRegressor model."""

    n_estimators: int = 100
    max_depth: int | None = None

class RandomForestRegressorConfig(ModelConfig):
    """Configuration for the RandomForestRegressor model."""

    running_config: RandomForestRegressorRunningConfig = RandomForestRegressorRunningConfig()
    hyperparameters: RandomForestRegressorHyperParameters = RandomForestRegressorHyperParameters()
    in_ports: dict[str, Port] = {
        "X": Port(arr_type=ArrayLikeEnum.NUMPY, data_category=DataCategoryEnum.MIXED, data_shape="batch features", desc="Input features for the regression task."),
        "y": Port(arr_type=ArrayLikeEnum.NUMPY, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch target_features", desc="Target values for the regression) task.")
    }
    out_ports: dict[str, Port] = {
        "pred": Port(arr_type=ArrayLikeEnum.NUMPY, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch target_features", desc="Predicted values for the regression task.")
    }

hyperparameter_space = {
    "n_estimators": ("choice", [50, 100, 200]),
    "max_depth": ("choice", [None, 10, 20, 30]),
}

class RandomForestRegressorNode(Model[np.ndarray, TabularDataContext, np.ndarray, TabularDataContext, dict[str, dict[str, list[str]] | dict[str, Any]]]):
    """RandomForestRegressor model node implementation."""

    metadata = RandomForestMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: RandomForestRegressorConfig) -> None:
        """Initialize the RandomForestRegressorNode with the given configuration."""
        super().__init__(config=config)
        self._model = SklearnRandomForestRegressor(
            n_estimators=self._config.hyperparameters.n_estimators,
            max_depth=self._config.hyperparameters.max_depth,
            criterion=self._config.running_config.criterion,
            random_state=self._config.running_config.random_state
        )
        self._params_target_context: dict[str, list[str]] = {}

    @property
    def _model_params(self) -> dict[str, Any]:
        return self._model.get_params()

    def fit(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> None:
        """Fit the RandomForestRegressor model with the given data."""
        # Here you would implement the logic to fit the RandomForestRegressor model using the input data.
        # This is just a placeholder implementation.
        X, _ = data["X"]
        y, y_ctx = data["y"]
        self._model.fit(X, y)
        self._params_target_context = y_ctx.dump_dict

    def predict(self, data: dict[str, tuple[np.ndarray, TabularDataContext]]) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
        """Predict using the RandomForestRegressor model with the given data."""
        # Here you would implement the logic to predict using the RandomForestRegressor model using the input data.
        # This is just a placeholder implementation.
        X, _ = data["X"]
        pred = self._model.predict(X)
        pred_ctx = tabular_context_from_dict_dump(self._params_target_context)  # We need to convert the target context to a tabular context for the output context, since the model is a regressor and the output is numerical data.
        return {"pred": (pred, pred_ctx)}

    def get_params(self) -> dict[str, dict[str, list[str]] | dict[str, Any]]:
        """Get the model parameters."""
        return {
            "model_params": self._model_params,
            "target_context": self._params_target_context
        }

    def set_params(self, params: dict[str, dict[str, list[str]] | dict[str, Any]]) -> None:
        """Set the model parameters."""
        self._params_target_context = params["target_context"]
        self._model.set_params(**params["model_params"])

