"""Correlation Feature Filter Node module."""
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


class CorrelationFilterMetadata(TransformMetadata):
    """Metadata for the CorrelationFilter Node."""

    node_name: str = "Correlation Feature Filter"
    input_type: str ="pd.DataFrame"
    output_type: str ="pd.DataFrame"
    description: str ="Filters out features with high correlation to other features."

class CorrelationFilterRunningConfig(TransformRunningConfig):
    """Runniong config for the Correlation Filter Node."""

    filtering_columns: list[str] | None = None  # If None, all columns will be filtered.

class CorrelationFilterHyperparameters(TransformHyperParameters):
    """Hyperparameters for the Correlation Filter Node."""

    threshold: float = 0.9
    corr_type: str = "pearson"  # Type of correlation to use. Can be "pearson", "spearman", or "kendall".

hyperparameter_space = {
    "threshold": ("float", {"min": 0, "max": 1}),
    "corr_type": ("categorical", {"choices": ["pearson", "spearman", "kendall"]}),
}

class CorrelationFilterConfig(TransformConfig):
    """Correlation Filter Config class."""

    running_config: CorrelationFilterRunningConfig = CorrelationFilterRunningConfig()
    hyperparameters: CorrelationFilterHyperparameters = CorrelationFilterHyperparameters()
    in_ports: dict[str, Port] = {"input": Port(type=pd.DataFrame, desc="Input data")}
    out_ports: dict[str, Port] = {"output": Port(type=pd.DataFrame, desc="output port")} # Pydantic already handles the deepcopy

class CorrelationFilter(TransformNode[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[str]]]):
    """Filters out nodes with high correlation to other features."""

    metadata = CorrelationFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: CorrelationFilterConfig) -> None:
        """Initialize the CorrelationFilter Node with the given configuration."""
        self._config = config
        self._params = None
        self._fitted = False

    def _get_filtered_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the requested numeric columns."""
        requested_columns = self._config.running_config.filtering_columns
        filtered_data_columns = filter_columns(data, requested_columns)
        return filter_dtypes(filtered_data_columns, requested_dtypes=["number"])

    def fit(self, data: dict[str, pd.DataFrame]) -> None:
        """Fit the CorrelationFilter Node with the given data."""
        df = data["input"]
        corr_type = self._config.hyperparameters.corr_type
        filtered_df = self._get_filtered_numeric_data(df)
        corr_matrix = filtered_df.corr(method=corr_type).abs() # type: ignore
        upper_tri = corr_matrix.where(
            pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool) # type: ignore
        )
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self._config.hyperparameters.threshold)]
        self._params = {"columns_to_filter": to_drop}
        self._fitted = True