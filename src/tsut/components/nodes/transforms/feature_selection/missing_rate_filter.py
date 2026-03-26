"""Missing Rate Filter for feature selection."""
# pyright: reportIncompatibleVariableOverride=false
import pandas as pd

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

    threshold: float = 0.95  # Threshold for missing rate to filter features. Default is 0.5 (i.e., drop features with more than 50% missing values).

class MissingRateFilterConfig(TransformConfig[MissingRateRunningConfig, MissingRateHyperParameters]):
    """Configuration for the MissingRateFilter TransformNode in the TSUT Framework."""

    running_config: MissingRateRunningConfig = MissingRateRunningConfig()
    hyperparameters: MissingRateHyperParameters = MissingRateHyperParameters()
    in_ports = {"input": Port(type=pd.DataFrame, desc="Input data")}
    out_ports = {"output": Port(type=pd.DataFrame, desc="output port")}

hyperparameter_space = {
    "threshold": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Threshold for missing rate to filter features."
    }
}

class MissingRateFilterNode(TransformNode[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[str]]]):
    """TransformNode implementation for filtering features based on their missing rate in the TSUT Framework."""

    metadata = MissingRateFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: MissingRateFilterConfig) -> None:
        self._config = config
        self._hyperparameters = config.hyperparameters
        self._params: dict[str, list[str]] = {} # This will hold the parameters after fitting, namely the columns to filter out.

    def fit(self, data: dict[str, pd.DataFrame]) -> None:
        """Fit the MissingRateFilterNode with the given data."""
        # Compute the missing rate for each column
        missing_rate = pd.Series(data["input"].isna().mean(), index=data["input"].columns)
        # Determine which columns to filter out based on the threshold
        columns_to_filter = pd.Series(missing_rate[missing_rate > self._hyperparameters.threshold]).index # Pyright being dumb, error will never happen here.
        self._params = {"columns_to_filter": list(columns_to_filter)}

    def transform(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Apply the missing rate filter to the given data."""
        columns_to_filter = self._params["columns_to_filter"]
        return {"output": data["input"].drop(columns=columns_to_filter)}
