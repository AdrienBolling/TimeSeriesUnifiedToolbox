"""Outlier Filter Transform for the TSUT Framework."""

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


class OutlierFilterMetadata(TransformMetadata):
    """Metadata for the OutlierFilter TransformNode."""

    node_name: str = "Outlier Filter"
    input_type: str = "pd.DataFrame"
    output_type: str = "pd.DataFrame"
    description: str = "Applies a strategy to outliers detected from using a method"

class OutlierFilterRunningConfig(TransformRunningConfig):
    """Running configuration for the OutlierFilter TransformNode."""

    filtering_columns: list[str] | None = None  # If None, all columns will be filtered.

class OutlierFilterHyperParameters(TransformHyperParameters):
    """Default Hyperparameters for the OutlierFilter TransformNode."""

    method: str = "iqr"
    strategy: str = "remove"
    threshold: float = 1.5

class OutlierFilterConfig(TransformConfig[OutlierFilterRunningConfig, OutlierFilterHyperParameters]):
    """Configuration for the OutlierFilter TransformNode."""

    running_config: OutlierFilterRunningConfig = OutlierFilterRunningConfig()
    hyperparameters: OutlierFilterHyperParameters = OutlierFilterHyperParameters()
    in_ports: dict[str, Port] = {"input": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch feature", desc="input port")}
    out_ports: dict[str, Port] = {"output": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch feature", desc="output port")}

hyperparameter_space = {
    "method": ("choice", ["iqr", "z_score"]),
    "strategy": ("choice", ["remove", "cap"]),
    "threshold": ("float", {"min": 0.0, "max": 10.0})
}

class OutlierFilter(
    TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, dict[str, dict[str, float]]]
):
    """Outlier Filter TransformNode for the TSUT Framework."""

    metadata = OutlierFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: OutlierFilterConfig) -> None:
        """Initialize the OutlierFilter with the given configuration."""
        self._config = config
        self._params: dict[str, dict[str, float]] = {}
        self._fitted = False

    # --- Helpers ---

    def _get_filtered_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the requested numeric columns.

        Assume all data is numerical.
        """
        requested_columns = self._config.running_config.filtering_columns
        return filter_columns(data, requested_columns)

    # --- Implement abstract methods from TransformNode ---

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Fit the OutlierFilter with the given data."""
        data_df, _ = data["input"]

        if self._config.hyperparameters.method == "iqr":
            self._fit_iqr_method(data_df)
        elif self._config.hyperparameters.method == "z_score":
            self._fit_z_score_method(data_df)
        else:
            raise ValueError(f"Unsupported method: {self._config.hyperparameters.method}")

        self._fitted = True

    def transform(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Apply the OutlierFilter to the given data."""
        if not self._fitted:
            raise RuntimeError("OutlierFilter must be fitted before calling transform().")
        outliers = self._detect_outliers(data["input"][0])
        return {"output": (self._handle_outliers(data["input"][0], outliers), data["input"][1])}

    def get_params(self) -> dict[str, dict[str, float]]:
        """Get the current parameters of the OutlierFilter."""
        return self._params

    def set_params(self, params: dict[str, dict[str, float]]) -> None:
        """Set the parameters of the OutlierFilter."""
        self._params = params
        self._fitted = True

    # --- Internal methods for outlier detection and handling ---

    def _detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers in the data based on the specified method."""
        if self._config.hyperparameters.method == "iqr":
            return self._iqr_method(data)
        if self._config.hyperparameters.method == "z_score":
            return self._z_score_method(data)
        raise ValueError(f"Unsupported method: {self._config.hyperparameters.method}")

    def _handle_outliers(self, data: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data based on the specified strategy."""
        strategy = self._config.hyperparameters.strategy

        if strategy == "remove":
            return self._remove_outliers(data, outliers)
        if strategy == "cap":
            return self._cap_outliers(data)

        raise ValueError(f"Unsupported strategy: {strategy}")

    # --- Fit methods ---

    def _fit_iqr_method(self, data: pd.DataFrame) -> None:
        """Fit the IQR method for outlier detection."""
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        q1 = pd.Series(filtered_data.quantile(0.25), index=columns).to_numpy()
        q3 = pd.Series(filtered_data.quantile(0.75), index=columns).to_numpy()
        iqr = q3 - q1

        self._params["q1"] = {str(col): float(q1_val) for col, q1_val in zip(columns, q1, strict=False)}
        self._params["q3"] = {str(col): float(q3_val) for col, q3_val in zip(columns, q3, strict=False)}
        self._params["iqr"] = {str(col): float(iqr_val) for col, iqr_val in zip(columns, iqr, strict=False)}

    def _fit_z_score_method(self, data: pd.DataFrame) -> None:
        """Fit the Z-score method for outlier detection."""
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        df_mean = pd.Series(filtered_data.mean()).to_numpy()
        df_std = pd.Series(filtered_data.std()).replace(0, np.nan).to_numpy()  # Replace std of 0 with NaN to avoid division by zero
        self._params["mean"] = {str(col): float(mean_val) for col, mean_val in zip(columns, df_mean, strict=False)}
        self._params["std"] = {str(col): float(std_val) for col, std_val in zip(columns, df_std, strict=False)}

    # --- Detection methods ---

    def _iqr_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using the IQR method.

        Returns a boolean DataFrame indicating if each value is an outlier.
        """
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        q1 = pd.Series(self._params["q1"], index=columns)
        q3 = pd.Series(self._params["q3"], index=columns)
        iqr = pd.Series(self._params["iqr"], index=columns)

        lower_bound = q1 - self._config.hyperparameters.threshold * iqr
        upper_bound = q3 + self._config.hyperparameters.threshold * iqr

        return filtered_data.lt(lower_bound, axis=1) | filtered_data.gt(upper_bound, axis=1)

    def _z_score_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using the Z-score method.

        Returns a boolean DataFrame indicating if each value is an outlier.
        """
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        mean = pd.Series(self._params["mean"], index=columns)
        std = pd.Series(self._params["std"], index=columns)

        z_scores = filtered_data.sub(mean, axis=1).div(std, axis=1)
        return z_scores.abs() > self._config.hyperparameters.threshold

    # --- Strategy methods ---

    def _remove_outliers(self, data: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data.

        If a sample is an outlier in any feature, it gets removed.
        """
        mask = ~outliers.any(axis=1)
        return pd.DataFrame(data.loc[mask], columns=data.columns)

    def _cap_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers in the selected numeric columns."""
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        if self._config.hyperparameters.method == "iqr":
            q1 = pd.Series(self._params["q1"], index=columns)
            q3 = pd.Series(self._params["q3"], index=columns)
            iqr = pd.Series(self._params["iqr"], index=columns)

            lower_bound = q1 - self._config.hyperparameters.threshold * iqr
            upper_bound = q3 + self._config.hyperparameters.threshold * iqr

        elif self._config.hyperparameters.method == "z_score":
            mean = pd.Series(self._params["mean"], index=columns)
            std = pd.Series(self._params["std"], index=columns)

            lower_bound = mean - self._config.hyperparameters.threshold * std
            upper_bound = mean + self._config.hyperparameters.threshold * std

        else:
            raise ValueError(f"Unsupported method: {self._config.hyperparameters.method}")

        for column in columns:
            data[column] = data[column].clip(
                lower=float(lower_bound[column]),
                upper=float(upper_bound[column]),
            )

        return data
