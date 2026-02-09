"""CSV Data Fetcher Node for the TSUT Framework."""

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field

from tsut.core.common.data.types import TabularData, TimeSeries
from tsut.core.nodes.base import Port
from tsut.core.nodes.source.base import DataSourceConfig, DataSourceNode


class CSVDataFetcherConfig(DataSourceConfig):
    """Configuration for CSV Data Fetcher Node."""

    file_path: str = Field(..., description="Path to the CSV file")
    data_type: Literal["timeseries", "tabular"] = Field(
        ..., description="Type of data to load"
    )
    time_column: str | None = Field(
        None, description="Column name for time index (for TimeSeries)"
    )
    value_columns: list[str] | None = Field(
        None, description="Columns to use as values/features"
    )
    batch_size: int = Field(1, description="Batch size for TimeSeries data")

    def __init__(self, **data: str | int | list[str] | None) -> None:
        """Initialize the CSVDataFetcherConfig."""
        super().__init__(**data)
        # Set output port based on data_type
        if self.data_type == "timeseries":
            self.out_ports = {
                "output": Port(
                    type=TimeSeries,
                    desc="Time series data loaded from CSV",
                    mode=["read"],
                )
            }
        else:
            self.out_ports = {
                "output": Port(
                    type=TabularData,
                    desc="Tabular data loaded from CSV",
                    mode=["read"],
                )
            }


class CSVDataFetcher(DataSourceNode[TimeSeries | TabularData]):
    """Node for fetching data from CSV files.

    This node reads CSV files and converts them to either TimeSeries or
    TabularData objects based on the configuration.

    Example:
        >>> config = CSVDataFetcherConfig(
        ...     file_path="data.csv",
        ...     data_type="timeseries",
        ...     time_column="timestamp",
        ...     value_columns=["value1", "value2"],
        ...     batch_size=1
        ... )
        >>> fetcher = CSVDataFetcher(config=config)
        >>> result = fetcher.node_transform({})
        >>> timeseries = result["output"]

    """

    def __init__(self, *, config: CSVDataFetcherConfig) -> None:
        """Initialize the CSV Data Fetcher Node.

        Args:
            config: Configuration for the CSV Data Fetcher

        """
        super().__init__(config=config)
        self.config = config

    def fetch_data(self) -> TimeSeries | TabularData:
        """Fetch data from CSV file.

        Returns:
            TimeSeries or TabularData depending on configuration

        Raises:
            FileNotFoundError: If the CSV file does not exist
            ValueError: If the data format is invalid or columns are missing
            pd.errors.ParserError: If CSV parsing fails

        """
        # Check if file exists
        file_path = Path(self.config.file_path)
        if not file_path.exists():
            msg = f"CSV file not found: {self.config.file_path}"
            raise FileNotFoundError(msg)

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
        except Exception as e:
            msg = f"Failed to read CSV file: {e}"
            raise ValueError(msg) from e

        # Validate file is not empty
        if df.empty:
            msg = "CSV file is empty"
            raise ValueError(msg)

        # Handle time series data
        if self.config.data_type == "timeseries":
            return self._create_timeseries(df)

        # Handle tabular data
        return self._create_tabular_data(df)

    def _create_timeseries(self, df: pd.DataFrame) -> TimeSeries:
        """Create TimeSeries from DataFrame.

        Args:
            df: DataFrame read from CSV

        Returns:
            TimeSeries object

        Raises:
            ValueError: If time_column is not specified or not found

        """
        if self.config.time_column is None:
            msg = "time_column must be specified for timeseries data_type"
            raise ValueError(msg)

        if self.config.time_column not in df.columns:
            msg = f"Time column '{self.config.time_column}' not found in CSV. Available columns: {list(df.columns)}"
            raise ValueError(msg)

        # Parse time column and set as index
        try:
            df[self.config.time_column] = pd.to_datetime(df[self.config.time_column])
        except Exception as e:
            msg = f"Failed to parse time column '{self.config.time_column}' as datetime: {e}"
            raise ValueError(msg) from e

        df = df.set_index(self.config.time_column)

        # Select value columns if specified
        if self.config.value_columns is not None:
            missing_cols = set(self.config.value_columns) - set(df.columns)
            if missing_cols:
                msg = f"Value columns {missing_cols} not found in CSV. Available columns: {list(df.columns)}"
                raise ValueError(msg)
            df = df[self.config.value_columns]

        # Convert to TimeSeries
        return TimeSeries.from_dataframe(
            df,
            batch_size=self.config.batch_size,
            value_columns=self.config.value_columns,
        )

    def _create_tabular_data(self, df: pd.DataFrame) -> TabularData:
        """Create TabularData from DataFrame.

        Args:
            df: DataFrame read from CSV

        Returns:
            TabularData object

        Raises:
            ValueError: If specified columns are not found

        """
        # Select value columns if specified
        if self.config.value_columns is not None:
            missing_cols = set(self.config.value_columns) - set(df.columns)
            if missing_cols:
                msg = f"Value columns {missing_cols} not found in CSV. Available columns: {list(df.columns)}"
                raise ValueError(msg)
            df = df[self.config.value_columns]

        # Convert to TabularData
        return TabularData.from_dataframe(df)
