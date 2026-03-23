"""Outlier Filter Transform for the TSUT Framework."""
import pandas as pd

from tsut.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
)


class OutlierFilterMetadata(TransformMetadata):
    """Metadata for the OutlierFilter TransformNode."""

    node_name: str = "Outlier Filter"
    input_type: str = "DataFrame"
    output_type: str = "DataFrame"
    description: str = "Applies a strategy to outliers detected from using a method"

class OutlierFilterRunningConfig(TransformConfig):
    """Running configuration for the OutlierFilter TransformNode."""

    filtering_columns: list[str] | None = None  # If None, all columns will be filtered.

class OutlierFilterHyperParameters(TransformHyperParameters):
    """Default Hyperparameters for the OutlierFilter TransformNode."""

    method: str = "iqr"
    strategy: str = "remove"
    threshold: float = 1.5

class OutlierFilterConfig(TransformConfig):
    """Configuration for the OutlierFilter TransformNode."""

    running_config: OutlierFilterRunningConfig = OutlierFilterRunningConfig()
    hyperparameters: OutlierFilterHyperParameters = OutlierFilterHyperParameters()  # Pyright is stupid, too lazy to fix, its just static type_checking

hyperparameter_space = {
    "method": ("choice", ["iqr", "z_score"]),
    "strategy": ("choice", ["remove", "cap"]),
    "threshold": ("float", {"min": 0.0, "max": 10.0})
}

class OutlierFilter(TransformNode[pd.DataFrame, pd.DataFrame, OutlierFilterHyperParameters]):
    """Outlier Filter TransformNode for the TSUT Framework."""

    metadata = OutlierFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: OutlierFilterConfig) -> None:
        """Initialize the OutlierFilter with the given configuration."""
        self._config = config

    # --- Implement abstract methods from TransformNode ---

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the OutlierFilter with the given data."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the OutlierFilter to the given data."""
        # This is just a placeholder implementation, you should replace it with the actual logic to filter outliers based on the specified method and strategy.
        return data

    # --- Internal methods for outlier detection and handling ---

    def _detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in the data based on the specified method."""
        # Placeholder for outlier detection logic
        return pd.DataFrame()  # Replace with actual outlier detection results

    def _handle_outliers(self, data: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data based on the specified strategy."""
        # Placeholder for outlier handling logic
        return data  # Replace with actual data after handling outliers

    # --- Methods

    def _iqr_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using the IQR method.

        Returns a boolean DataFrame indicating if each value is an outlier.
        """
        requested_columns = self._config.running_config.filtering_columns

        if requested_columns is None:
            selected_columns = data.select_dtypes(include="number").columns.tolist()
        else:
            missing_columns = [col for col in requested_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(
                    f"The following filtering_columns are not in the DataFrame: {missing_columns}"
                )

            non_numeric_columns = [
                col for col in requested_columns if not pd.api.types.is_numeric_dtype(data[col])
            ]
            if non_numeric_columns:
                raise ValueError(
                    f"The following filtering_columns are not numeric and cannot be used "
                    f"with the IQR method: {non_numeric_columns}"
                )

            selected_columns = requested_columns

        if not selected_columns:
            raise ValueError("No numeric columns available for outlier filtering.")

        filtered_data = data[selected_columns]

        q1 = filtered_data.quantile(0.25)
        q3 = filtered_data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - self._config.hyperparameters.threshold * iqr
        upper_bound = q3 + self._config.hyperparameters.threshold * iqr

        outliers = (filtered_data < lower_bound) | (filtered_data > upper_bound)

        return outliers
