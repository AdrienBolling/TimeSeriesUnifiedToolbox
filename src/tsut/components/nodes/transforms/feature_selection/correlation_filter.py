"""Correlation Feature Filter Node module."""
import pandas as pd

from tsut.components.utils.dataframe import filter_columns, filter_dtypes
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
    in_ports: dict[str, Port] = {"input": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch features", desc="input port")} # Pydantic already handles the deepcopy
    out_ports: dict[str, Port] = {"output": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch features", desc="output port")} # Pydantic already handles the deepcopy

class CorrelationFilter(TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, dict[str, list[str]]]):
    """Filters out nodes with high correlation to other features."""

    metadata = CorrelationFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: CorrelationFilterConfig) -> None:
        """Initialize the CorrelationFilter Node with the given configuration."""
        self._config = config
        self._params: dict[str, list[str]] = {}  # To store the columns to filter after fitting. This is what will be used during transform to actually filter the columns.

    def _get_filtered_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the requested numeric columns."""
        requested_columns = self._config.running_config.filtering_columns
        filtered_data_columns = filter_columns(data, requested_columns)
        return filter_dtypes(filtered_data_columns, requested_dtypes=["number"])

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Fit the CorrelationFilter Node with the given data."""
        df, _ = data["input"]
        corr_type = self._config.hyperparameters.corr_type
        filtered_df = self._get_filtered_numeric_data(df)
        corr_matrix = filtered_df.corr(method=corr_type).abs() # type: ignore # Error due to pandas typing but verified there's no issue
        upper_tri = corr_matrix.where(
            pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool) # type: ignore
        )
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self._config.hyperparameters.threshold)]
        self._params = {"columns_to_filter": to_drop}

    def transform(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply the correlation filter to the given data."""
        df, context = data["input"]
        columns_to_drop = self._params["columns_to_filter"]  # Will never miss because of the check in the base Transform API.
        filtered_df = df.drop(columns=columns_to_drop)
        context.remove_columns(columns_to_drop)
        return {"output": (filtered_df, context)}

    def get_params(self) -> dict[str, list[str]]:
        """Return the parameters of the CorrelationFilter Node."""
        return self._params

    def set_params(self, params: dict[str, list[str]]) -> None:
        """Set the parameters of the CorrelationFilter Node."""
        self._params = params
