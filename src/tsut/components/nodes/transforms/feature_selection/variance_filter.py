"""Variance Filter for features"""

import numpy as np
import pandas as pd

from tsut.components.utils.dataframe import filter_columns, filter_dtypes
from tsut.core.nodes.node import Port
from tsut.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)


class VarianceFilterMetadata(TransformMetadata):
    """Metadata for the VarianceFilter Node."""

    node_name: str = "Variance Feature Filter"
    input_type: str ="pd.DataFrame"
    output_type: str ="pd.DataFrame"
    description: str ="Filters out features with near 0 variance."

class VarianceFilterRunningConfig(TransformRunningConfig):
    """Runniong config for the Variance Filter Node."""

    filtering_columns: list[str] | None = None  # If None, all columns will be filtered.

class VarianceFilterHyperparameters(TransformHyperParameters):
    """Hyperparameters for the Variance Filter Node."""

    threshold: float = 0.1

hyperparameter_space = {
    "threshold": ("float", {"min": 0, "max": np.inf}),
}

class VarianceFilterConfig(TransformConfig):
    """Variance Filter Config class."""

    running_config: VarianceFilterRunningConfig = VarianceFilterRunningConfig()
    hyperparameters: VarianceFilterHyperparameters = VarianceFilterHyperparameters()
    in_ports: dict[str, Port] = {"input": Port(type=pd.DataFrame, desc="Input data")}
    out_ports: dict[str, Port] = {"output": Port(type=pd.DataFrame, desc="output port")} # Pydantic already handles the deepcopy


class VarianceFilter(TransformNode[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[str]]]):
    """Filters out nodes with near-zero variance."""

    metadata = VarianceFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: VarianceFilterConfig) -> None:
        """Initialize the VarianceFilter Node with the given configuration."""
        self._config = config

    def _get_filtered_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the requested numeric columns."""
        requested_columns = self._config.running_config.filtering_columns
        filtered_data_columns = filter_columns(data, requested_columns)
        return filter_dtypes(filtered_data_columns, requested_dtypes=["number"])

    def fit(self, data: dict[str, pd.DataFrame]) -> None:
        """Fit the VarianceFilter with the given data."""
        # Isolate the columns to filter based on the running config
        data_df = data["input"]
        data_to_filter = self._get_filtered_numeric_data(data_df)
        # Compute the variance for each column
        variance = pd.Series(data_to_filter.var(), index=data_to_filter.columns)
        # Determine which columns to filter out based on the threshold
        columns_to_filter = pd.Series(variance[variance < self._config.hyperparameters.threshold]).index
        self._params = {"columns_to_filter": [str(col) for col in list(columns_to_filter)]}

        # Check the validity of the columns to filter
        if len(columns_to_filter) == 0:
            raise ValueError("No columns to filter based on the given threshold. Consider lowering the threshold or checking the data.")

    def transform(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Transform the data by filtering out the columns with near-zero variance."""
        columns_to_filter = self._params["columns_to_filter"]
        invert_columns_to_filter = [col for col in data["input"].columns if col not in columns_to_filter]
        transformed_data = filter_columns(data["input"], invert_columns_to_filter)
        return {"output": transformed_data}

    def get_params(self) -> dict[str, list[str]]:
        """Get the current parameters of the VarianceFilter, namely the columns that are being filtered out."""
        return self._params

    def set_params(self, params: dict[str, list[str]]) -> None:
        """Set the parameters of the VarianceFilter. This can be used to set the columns to filter out directly, without fitting."""
        self._params = params




