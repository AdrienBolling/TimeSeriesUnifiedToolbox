"""Modulefor simple numerical imputation node."""

import numpy as np
import pandas as pd

from tsut.components.utils.dataframe import filter_columns
from tsut.core.common.data.data import ArrayLikeEnum, DataCategoryEnum
from tsut.core.common.data.tabular_data import TabularDataContext
from tsut.core.nodes.node import Port
from tsut.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)


class SimpleNumericalImputationMetadata(TransformMetadata):
    """Metadata for the SimpleNumericalImputation TransformNode in a TSUT Pipeline."""

    node_name: str = "SimpleNumericalImputation"
    input_type: str = "pd.DataFrame"
    output_type: str = "pd.DataFrame"
    description: str = "Impute missing values in numerical features using a simple strategy (mean, median, or mode)."

class SimpleNumericalImputationRunningConfig(TransformRunningConfig):
    """Running configuration for the SimpleNumericalImputation TransformNode in the TSUT Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the transform itself, but rather for how to run it.
    For example, in some transforms, this could be very specific parameters such as the backend to use for computations etc.
    """

    imputation_columns: list[str] | None = None  # If None, all numeric columns will be imputed.

class SimpleNumericalImputationHyperParameters(TransformHyperParameters):
    """Hyperparameters for the SimpleNumericalImputation TransformNode in the TSUT Framework.

    This will usually be used for parameters that are relevant for the definition of the transform itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in some transforms, this could be the window size for a rolling window transform, etc.
    """

    strategy: str = "mean"  # Strategy for imputation. Can be "mean", "median", or "constant".
    value: float = 0.0  # Only used if strategy is "constant". The value to use for imputation.

class SimpleNumericalImputationConfig(TransformConfig[SimpleNumericalImputationRunningConfig, SimpleNumericalImputationHyperParameters]):
    """Configuration for the SimpleNumericalImputation TransformNode in the TSUT Framework."""

    running_config: SimpleNumericalImputationRunningConfig = SimpleNumericalImputationRunningConfig()
    hyperparameters: SimpleNumericalImputationHyperParameters = SimpleNumericalImputationHyperParameters()
    in_ports: dict[str, Port] = {"input": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch feature", desc="Input DataFrame with numerical features to impute")}
    out_ports: dict[str, Port] = {"output": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch feature", desc="Output DataFrame with imputed values for numerical features")}

hyperparameter_space = {
    "strategy": {
        "choice": ["mean", "median", "constant"],
    "value": ("float", {"min": -np.inf, "max": np.inf}),  # Only used if strategy is "constant"

    }
}

class SimpleNumericalImputationNode(TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, dict[str, dict[str, float]]]):
    """TransformNode implementation for simple numerical imputation in the TSUT Framework."""

    metadata = SimpleNumericalImputationMetadata()

    def __init__(self, *, config: SimpleNumericalImputationConfig) -> None:
        """Initialize the SimpleNumericalImputationNode with the given configuration."""
        self._config = config
        self._params = {"imputation_values": {}} # This will hold the imputation values after fitting, which will be used during transform to actually impute the missing values.

    def _get_filtered_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the requested numeric columns.

        Assume that the data only contains numerical columns.
        """
        if self._config.running_config.imputation_columns is not None:
            data = filter_columns(data, self._config.running_config.imputation_columns)
        return data

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Fit the imputation values based on the input data and the specified strategy."""
        df, ctx = data["input"]
        df = self._get_filtered_data(df)

        imputation_values = {}
        for col in df.columns:
            if self._config.hyperparameters.strategy == "mean":
                imputation_values[col] = df[col].mean()
            elif self._config.hyperparameters.strategy == "median":
                imputation_values[col] = df[col].median()
            elif self._config.hyperparameters.strategy == "constant":
                imputation_values[col] = self._config.hyperparameters.value
            else:
                raise ValueError(f"Invalid imputation strategy: {self._config.hyperparameters.strategy}")

        self._params["imputation_values"] = imputation_values

    def transform(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Impute the missing values in the input DataFrame based on the fitted imputation values."""
        df, ctx = data["input"]
        df = self._get_filtered_data(df)

        imputed_df = df.fillna(self._params["imputation_values"])

        return {"output": (imputed_df, ctx)}

    def get_params(self) -> dict[str, dict[str, float]]:
        """Get the current imputation values."""
        return self._params

    def set_params(self, params: dict[str, dict[str, float]]) -> None:
        """Set the imputation values."""
        self._params = params
