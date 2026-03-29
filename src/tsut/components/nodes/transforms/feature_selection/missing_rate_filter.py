"""Missing Rate Filter for feature selection."""

import pandas as pd

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


class MissingRateFilterMetadata(TransformMetadata):
    """Metadata for the MissingRateFilter TransformNode in a TSUT Pipeline."""

    node_name: str = "MissingRateFilter"
    input_type: str = "pd.DataFrame"
    output_type: str = "pd.DataFrame"
    description: str = "Filter features based on their missing rate. This transform will drop the features that have a missing rate higher than a given threshold."

class MissingRateRunningConfig(TransformRunningConfig):
    """Running configuration for the MissingRateFilter TransformNode in the TSUT Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the transform itself, but rather for how to run it.
    For example, in some transforms, this could be very specific parameters such as the backend to use for computations etc.
    """

class MissingRateHyperParameters(TransformHyperParameters):
    """Hyperparameters for the MissingRateFilter TransformNode in the TSUT Framework.

    This will usually be used for parameters that are relevant for the definition of the transform itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in some transforms, this could be the window size for a rolling window transform, etc.
    """

    threshold: float = 0.5  # Threshold for missing rate to filter features. Default is 0.5 (i.e., drop features with more than 50% missing values).

class MissingRateFilterConfig(TransformConfig[MissingRateRunningConfig, MissingRateHyperParameters]):
    """Configuration for the MissingRateFilter TransformNode in the TSUT Framework."""

    running_config: MissingRateRunningConfig = MissingRateRunningConfig()
    hyperparameters: MissingRateHyperParameters = MissingRateHyperParameters()
    in_ports: dict[str, Port] = {"input": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch feature", desc="Input DataFrame with features to filter based on missing rate")}
    out_ports: dict[str, Port] = {"output": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch feature", desc="Output DataFrame with features filtered based on missing rate")}

hyperparameter_space = {
    "threshold": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Threshold for missing rate to filter features."
    }
}

class MissingRateFilterNode(TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, dict[str, list[str]]]):
    """TransformNode implementation for filtering features based on their missing rate in the TSUT Framework."""

    metadata = MissingRateFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: MissingRateFilterConfig) -> None:
        """Initialize the MissingRateFilterNode with the given configuration."""
        self._config = config
        self._hyperparameters = config.hyperparameters
        self._params: dict[str, list[str]] = {} # This will hold the parameters after fitting, namely the columns to filter out.

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Fit the MissingRateFilterNode with the given data."""
        # Compute the missing rate for each column
        df, _ = data["input"]
        missing_rate = pd.Series(df.isna().mean(), index=df.columns)
        # Determine which columns to filter out based on the threshold
        columns_to_filter = pd.Series(missing_rate[missing_rate > self._hyperparameters.threshold]).index # Pyright being dumb, error will never happen here.
        self._params = {"columns_to_filter": list(columns_to_filter)}

    def transform(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply the missing rate filter to the given data."""
        df, context = data["input"]
        columns_to_filter = self._params["columns_to_filter"]
        filtered_df = df.drop(columns=columns_to_filter)
        context.remove_columns(columns_to_filter)
        return {"output": (filtered_df, context)}

    def get_params(self) -> dict[str, list[str]]:
        """Return the parameters of the MissingRateFilterNode, namely the columns to filter out."""
        return self._params

    def set_params(self, params: dict[str, list[str]]) -> None:
        """Set the parameters of the MissingRateFilterNode, namely the columns to filter out."""
        self._params = params
