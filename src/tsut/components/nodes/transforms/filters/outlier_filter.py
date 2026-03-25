"""Outlier Filter Transform for the TSUT Framework."""

import numpy as np
import pandas as pd

from tsut.components.utils.dataframe import filter_columns, filter_dtypes
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
    in_ports = {"input": Port(type=pd.DataFrame, desc="Input data")}
    out_ports = {"output": Port(type=pd.DataFrame, desc="output port")}

hyperparameter_space = {
    "method": ("choice", ["iqr", "z_score"]),
    "strategy": ("choice", ["remove", "cap"]),
    "threshold": ("float", {"min": 0.0, "max": 10.0})
}

class OutlierFilter(
    TransformNode[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, float]]]
):
    """Outlier Filter TransformNode for the TSUT Framework."""

    metadata = OutlierFilterMetadata()
    hyperparameter_space = hyperparameter_space

    def __init__(self, *, config: OutlierFilterConfig) -> None:
        """Initialize the OutlierFilter with the given configuration."""
        self._config = config
        self._params: dict[str, dict[str, float]] = {}
        self._fitted = False
        self._validate_hyperparameters()

    # --- Validation of hyperparameters and running configuration ---

    def _validate_hyperparameters(self) -> None:
        """Validate the hyperparameters of the OutlierFilter."""
        if self._config.hyperparameters.method not in hyperparameter_space["method"][1]:
            raise ValueError(
                f"Invalid method: {self._config.hyperparameters.method}. "
                f"Must be one of {hyperparameter_space['method'][1]}"
            )

        if self._config.hyperparameters.strategy not in hyperparameter_space["strategy"][1]:
            raise ValueError(
                f"Invalid strategy: {self._config.hyperparameters.strategy}. "
                f"Must be one of {hyperparameter_space['strategy'][1]}"
            )

        threshold = self._config.hyperparameters.threshold
        threshold_min = hyperparameter_space["threshold"][1]["min"]
        threshold_max = hyperparameter_space["threshold"][1]["max"]
        if not (threshold_min <= threshold <= threshold_max):
            raise ValueError(
                f"Invalid threshold: {threshold}. "
                f"Must be between {threshold_min} and {threshold_max}"
            )

    # --- Helpers ---

    def _get_filtered_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return only the requested numeric columns."""
        requested_columns = self._config.running_config.filtering_columns
        filtered_data_columns = filter_columns(data, requested_columns)
        filtered_data_dtypes = filter_dtypes(filtered_data_columns, requested_dtypes=["number"])
        return filtered_data_dtypes

    def _series_to_param_dict(self, values: pd.Series | float, columns: pd.Index) -> dict[str, float]:
        """Convert scalar/Series outputs into a serializable dict keyed by column name."""
        if np.isscalar(values):
            if len(columns) != 1:
                raise ValueError(
                    "Received scalar statistic for multiple columns, which is ambiguous."
                )
            return {str(columns[0]): float(values)}

        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=columns)

        return {str(col): float(values[col]) for col in columns}

    def _param_dict_to_series(self, name: str, columns: pd.Index) -> pd.Series:
        """Convert stored dict params back to a Series aligned on current columns."""
        if name not in self._params:
            raise ValueError(f"Missing parameter '{name}'. Did you call fit()?")

        param_values = self._params[name]
        missing_columns = [str(col) for col in columns if str(col) not in param_values]
        if missing_columns:
            raise ValueError(
                f"Stored parameters for '{name}' are missing columns: {missing_columns}"
            )

        return pd.Series(
            {col: float(param_values[str(col)]) for col in columns},
            index=columns,
            dtype=float,
        )

    # --- Implement abstract methods from TransformNode ---

    def fit(self, data: dict[str, pd.DataFrame]) -> None:
        """Fit the OutlierFilter with the given data."""
        self._validate_hyperparameters()
        self._params = {}
        data_df = data["input"]

        if self._config.hyperparameters.method == "iqr":
            self._fit_iqr_method(data_df)
        elif self._config.hyperparameters.method == "z_score":
            self._fit_z_score_method(data_df)
        else:
            raise ValueError(f"Unsupported method: {self._config.hyperparameters.method}")

        self._fitted = True

    def transform(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Apply the OutlierFilter to the given data."""
        if not self._fitted:
            raise RuntimeError("OutlierFilter must be fitted before calling transform().")

        outliers = self._detect_outliers(data["input"])
        return {"output": self._handle_outliers(data["input"].copy(), outliers)}

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

        q1 = filtered_data.quantile(0.25)
        q3 = filtered_data.quantile(0.75)
        iqr = q3 - q1

        self._params["q1"] = self._series_to_param_dict(q1, columns)
        self._params["q3"] = self._series_to_param_dict(q3, columns)
        self._params["iqr"] = self._series_to_param_dict(iqr, columns)

    def _fit_z_score_method(self, data: pd.DataFrame) -> None:
        """Fit the Z-score method for outlier detection."""
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        df_mean = filtered_data.mean()
        df_std = pd.Series(filtered_data.std()).replace(0, np.nan)

        self._params["mean"] = self._series_to_param_dict(df_mean, columns)
        self._params["std"] = self._series_to_param_dict(df_std, columns)

    # --- Detection methods ---

    def _iqr_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using the IQR method.

        Returns a boolean DataFrame indicating if each value is an outlier.
        """
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        q1 = self._param_dict_to_series("q1", columns)
        q3 = self._param_dict_to_series("q3", columns)
        iqr = self._param_dict_to_series("iqr", columns)

        lower_bound = q1 - self._config.hyperparameters.threshold * iqr
        upper_bound = q3 + self._config.hyperparameters.threshold * iqr

        return filtered_data.lt(lower_bound, axis=1) | filtered_data.gt(upper_bound, axis=1)

    def _z_score_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using the Z-score method.

        Returns a boolean DataFrame indicating if each value is an outlier.
        """
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        mean = self._param_dict_to_series("mean", columns)
        std = self._param_dict_to_series("std", columns)

        z_scores = filtered_data.sub(mean, axis=1).div(std, axis=1)
        return z_scores.abs() > self._config.hyperparameters.threshold

    # --- Strategy methods ---

    def _remove_outliers(self, data: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data.

        If a sample is an outlier in any feature, it gets removed.
        """
        mask = ~outliers.any(axis=1)
        return data.loc[mask]

    def _cap_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers in the selected numeric columns."""
        filtered_data = self._get_filtered_numeric_data(data)
        columns = filtered_data.columns

        if self._config.hyperparameters.method == "iqr":
            q1 = self._param_dict_to_series("q1", columns)
            q3 = self._param_dict_to_series("q3", columns)
            iqr = self._param_dict_to_series("iqr", columns)

            lower_bound = q1 - self._config.hyperparameters.threshold * iqr
            upper_bound = q3 + self._config.hyperparameters.threshold * iqr

        elif self._config.hyperparameters.method == "z_score":
            mean = self._param_dict_to_series("mean", columns)
            std = self._param_dict_to_series("std", columns)

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
