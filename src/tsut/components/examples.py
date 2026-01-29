"""This module serves as an example of implementation using the TSUT framework"""

from tsut.core.nodes.models.base import Model, ModelConfig
from tsut.core.nodes.models.mixins.regressors import RegressorMixin

from sklearn.linear_model import LinearRegression
    import numpy as np

class LinearRegressorModelConfig(ModelConfig):
    fit_intercept: bool = True
    copy_X: bool = True
    tol: float = 1e-4
    n_jobs: int | None = None
    positive: bool = False

    in_ports: dict[str, any] = {
        "X": {"type": np.ndarray, "desc": "Input features", "mode": ["fit", "predict"]},
        "y": {"type": np.ndarray, "desc": "Target values", "mode": ["fit"]},
    }
    out_ports: dict[str, any] = {
        "predictions": {"type": np.ndarray, "desc": "Predicted values", "mode": ["predict"]},
    }

class LinearRegressorModel(Model, RegressorMixin):
    
    def __init__(self, *, config: LinearRegressorModelConfig) -> None:
        super().__init__(config=config)
        self.model = LinearRegression(
            fit_intercept=config.fit_intercept,
            copy_X=config.copy_X,
            tol=config.tol,
            n_jobs=config.n_jobs,
            positive=config.positive,
        )

    def fit(self, data: dict[str, any]) -> None:
        self.model.fit(data["X"], data["y"])

    def predict(self, data: dict[str, any]) -> dict[str, any]:
        predictions = self.model.predict(data["X"])
        return {"predictions": predictions}

    def get_params(self) -> dict[str, any]:
        return self.model.get_params()

    def restore_params(self, params: dict[str, any]) -> None:
        for param, value in params.items():
            setattr(self.model, param, value)
