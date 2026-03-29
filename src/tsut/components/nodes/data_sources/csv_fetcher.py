"""CSV Data Fetcher Node module"""

import json

import pandas as pd

from tsut.core.common.data.data import ArrayLikeEnum, DataCategoryEnum
from tsut.core.common.data.tabular_data import (
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from tsut.core.nodes.data_source.data_source import (
    DataSourceConfig,
    DataSourceMetadata,
    DataSourceNode,
    DataSourceRunningConfig,
)
from tsut.core.nodes.node import Port


class CSVFetcherMetadata(DataSourceMetadata):
    """Metadata for the CSVFetcher data source node."""

    name: str = "CSVFetcher"
    description: str = "Data source node that fetches data from a CSV file."

class CSVFetcherRunningConfig(DataSourceRunningConfig):
    """Running configuration for the CSVFetcher data source node."""

    data_file_path: str = "data.csv"  # Path to the CSV file to fetch data from.
    context_file_path: str = "data_context.json"  #  path to a context file that contains additional information about the data (e.g., feature types, target variable name, etc.). This can be used to provide additional context for the fetched data, which can be useful for downstream nodes in the pipeline.

    target_file_path: str = "target.csv"  #  path to a target file that contains the target variable values. This can be used to provide the target variable values for supervised learning tasks, which can be useful for downstream nodes in the pipeline that require the target variable values (e.g., a model node that needs the target variable values for training).
    target_context_file_path: str = "target_context.json"  #  path to a target context file that contains additional information about the target variable (e.g., target variable type, etc.). This can be used to provide additional context for the target variable, which can be useful for downstream nodes in the pipeline that require the target variable values (e.g., a model node that needs the target variable values for training).


class CSVFetcherConfig(DataSourceConfig[CSVFetcherRunningConfig]):
    """Configuration for the CSVFetcher data source node."""

    running_config: CSVFetcherRunningConfig = CSVFetcherRunningConfig()
    in_ports: dict[str, Port] = {}  # No input ports for a data source node.
    out_ports: dict[str, Port] = {
        "data": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch features", desc="Fetched data from the CSV file."),
        "target": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.CATEGORICAL, data_shape="batch target_features", desc="Target values from the CSV file.")
    }

class CSVFetcherNode(DataSourceNode[pd.DataFrame, TabularDataContext]):
    """CSVFetcher data source node implementation."""

    metadata = CSVFetcherMetadata()

    def __init__(self, *, config: CSVFetcherConfig) -> None:
        """Initialize the CSVFetcherNode with the given configuration."""
        super().__init__(config=config)

    def setup_source(self) -> None:
        """Set up the data source (e.g., establish connections, load resources)."""

    def fetch_data(self) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Fetch data from the CSV file."""
        data_df = pd.read_csv(self._config.running_config.data_file_path)
        target_df = pd.read_csv(self._config.running_config.target_file_path)

        with open(self._config.running_config.context_file_path) as f:
            data_context_dict = json.load(f)
        data_context = tabular_context_from_dict_dump(data_context_dict)
        with open(self._config.running_config.target_context_file_path) as f:
            target_context_dict = json.load(f)
        target_context = tabular_context_from_dict_dump(target_context_dict)

        return {
            "data": (data_df, data_context),
            "target": (target_df, target_context)
        }
