"""RemoveCategorical node module."""

import pandas as pd

from tsut.core.common.data.data import INVERSE_DATA_CATEGORY_MAPPING, ArrayLikeEnum, DataCategoryEnum
from tsut.core.common.data.tabular_data import TabularDataContext
from tsut.core.nodes.node import Port
from tsut.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)


class RemoveCategoricalMetadata(TransformMetadata):
    """Metadata for the RemoveCategorical TransformNode in a TSUT Pipeline."""

    node_name: str = "RemoveCategorical"
    description: str = "Remove categorical features from the input DataFrame. This transform will drop all the features that are categorized as categorical in the input context."

class RemoveCategoricalRunningConfig(TransformRunningConfig):
    """Running configuration for the RemoveCategorical TransformNode in the TSUT Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the transform itself, but rather for how to run it.
    For example, in some transforms, this could be very specific parameters such as the backend to use for computations etc.
    """

class RemoveCategoricalHyperParameters(TransformHyperParameters):
    """Hyperparameters for the RemoveCategorical TransformNode in the TSUT Framework.

    This will usually be used for parameters that are relevant for the definition of the transform itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in some transforms, this could be the window size for a rolling window transform, etc.
    """

class RemoveCategoricalConfig(TransformConfig[RemoveCategoricalRunningConfig, RemoveCategoricalHyperParameters]):
    """Configuration for the RemoveCategorical TransformNode in the TSUT Framework."""

    running_config: RemoveCategoricalRunningConfig = RemoveCategoricalRunningConfig()
    hyperparameters: RemoveCategoricalHyperParameters = RemoveCategoricalHyperParameters()
    in_ports: dict[str, Port] = {"input": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch feature", desc="Input DataFrame with features to filter based on missing rate")}
    out_ports: dict[str, Port] = {"output": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch feature", desc="Output DataFrame with categorical features removed")}

class RemoveCategoricalNode(TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, dict[str, list[str]]]):
    """TransformNode implementation for removing categorical features from the input DataFrame in the TSUT Framework."""

    metadata = RemoveCategoricalMetadata()

    def __init__(self, *, config: RemoveCategoricalConfig) -> None:
        """Initialize the RemoveCategoricalNode with the given configuration."""
        self._config = config
        self._params: dict[str, list[str]] = {}

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Fit the transform with the given data."""
        # Find the node with the context.categories that are categorical, and store the names of the features to remove in self._features_to_remove. This will be used in the transform method to actually remove the categorical features from the input DataFrame.
        input_df, input_ctx = data["input"]
        categorical_features = [feature for feature, category in zip(input_df.columns, input_ctx.categories) if str(INVERSE_DATA_CATEGORY_MAPPING[category]) == "categorical_data"]
        self._params["features_to_remove"] = categorical_features


    def transform(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply the transform to the given data."""
        # This is just a placeholder implementation. In a real implementation, you would probably want to check the input context to identify which features are categorical and should be removed, and then return the transformed DataFrame with the categorical features removed.
        input_df, input_ctx = data["input"]
        output_df = pd.DataFrame(input_df.drop(columns=self._params["features_to_remove"]))
        input_ctx.remove_columns(self._params["features_to_remove"])
        return {"output": (output_df, input_ctx)}

    def get_params(self) -> dict[str, list[str]]:
        """Get the current parameters of the transform."""
        return self._params

    def set_params(self, params: dict[str, list[str]]) -> None:
        """Set the parameters of the transform."""
        self._params = params
