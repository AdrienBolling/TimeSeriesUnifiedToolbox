"""Define the base Data types for the TSUT Framework."""

from typing import Any, NamedTuple, TypeVar

import numpy as np
import pandas as pd
from jaxtyping import Float
from pydantic import BaseModel

D = TypeVar("D", bound="Data")
K = TypeVar("K")


type ContextData = (
    dict[str, Any] | BaseModel | NamedTuple
)  # Abstarct type to alias all types used to pass context data. (Such as runtime information, etc.)

type ConfigData = (
    dict[str, Any] | BaseModel | NamedTuple
)  # Abstract type to alias all types used to pass configuraiton data. (Such as hyperparameters, etc.)

# Constants for internal data shape validation
_TIMESERIES_NDIM = 4  # (batch, dimension, timestep, value)
_TABULAR_NDIM = 2  # (batch, feature)


class Data:
    """Base class for all data types in the TSUT Framework."""

    # INFO : For now there isn't any specifics that need to be defined across all data types. It is intentional, this type is only used to ensure all inputs/outputs come from the TSUT library.


class TimeSeries(Data):
    """Class representing time series data structured as (batch, dimension, timestep, value).

    The implementation is highly inspired from the Darts TimeSeries class. It is intentionally more lightweight however.
    All capabilities related to auto-pre-treatment have been removed as these are intended to be integrated as pipeline Nodes.

    The data is stored as a 4D numpy array with dimensions:
    - batch: Number of independent time series samples in a batch
    - dimension: Number of features/variables (1 for univariate, >1 for multivariate)
    - timestep: Number of time points in the series
    - value: Typically 1, but can be used for additional value dimensions

    This structure supports both univariate and multivariate time series with batched processing across models.
    """

    def __init__(
        self,
        times: pd.DatetimeIndex | pd.RangeIndex | pd.Index,
        values: Float[np.ndarray, "batch dimension timestep value"],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a 'TimeSeries' from a time index 'times' and a numpy array of 'values'.

        Args:
            times: Time index for the series (DatetimeIndex, RangeIndex, or Index)
            values: 4D numpy array structured as (batch, dimension, timestep, value)
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If the time index length doesn't match the timestep dimension
            ValueError: If values is not a 4D array

        """
        if values.ndim != _TIMESERIES_NDIM:
            msg = f"Values must be a {_TIMESERIES_NDIM}D array (batch, dimension, timestep, value), got {values.ndim}D array"
            raise ValueError(msg)

        if len(times) != values.shape[2]:
            msg = f"Time index length ({len(times)}) must match timestep dimension ({values.shape[2]})"
            raise ValueError(msg)

        self._times = times
        self._values = values
        self._metadata = metadata or {}

    @property
    def times(self) -> pd.DatetimeIndex | pd.RangeIndex | pd.Index:
        """Return the time index."""
        return self._times

    @property
    def values(self) -> Float[np.ndarray, "batch dimension timestep value"]:
        """Return the values array.

        Returns:
            4D numpy array with shape (batch, dimension, timestep, value).

        """
        return self._values

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata dictionary."""
        return self._metadata

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return the shape of the values array as (batch, dimension, timestep, value)."""
        return self._values.shape

    @property
    def n_batches(self) -> int:
        """Return the number of batches."""
        return self._values.shape[0]

    @property
    def n_dimensions(self) -> int:
        """Return the number of dimensions/features."""
        return self._values.shape[1]

    @property
    def n_timesteps(self) -> int:
        """Return the number of timesteps."""
        return self._values.shape[2]

    @property
    def n_values(self) -> int:
        """Return the size of the value dimension."""
        return self._values.shape[3]

    @property
    def is_univariate(self) -> bool:
        """Return True if the time series is univariate (single dimension)."""
        return self.n_dimensions == 1

    @property
    def is_multivariate(self) -> bool:
        """Return True if the time series is multivariate (multiple dimensions)."""
        return self.n_dimensions > 1

    # Conversion methods: FROM common formats

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        batch_size: int = 1,
        value_columns: list[str] | None = None,
    ) -> "TimeSeries":
        """Create a TimeSeries from a pandas DataFrame.

        Args:
            df: DataFrame with time index and value columns
            batch_size: Number of samples in batch (default: 1)
            value_columns: List of column names to use as dimensions (default: all columns)

        Returns:
            TimeSeries instance

        Raises:
            ValueError: If DataFrame index is not temporal

        """
        if not isinstance(df.index, (pd.DatetimeIndex, pd.RangeIndex)):
            msg = "DataFrame must have DatetimeIndex or RangeIndex"
            raise TypeError(msg)

        if value_columns is None:
            value_columns = list(df.columns)

        # Extract values: (timesteps, dimensions)
        values_2d = df[value_columns].to_numpy()
        n_timesteps, n_dimensions = values_2d.shape

        # Reshape to (batch, dimension, timestep, value)
        values_4d = values_2d.T.reshape(batch_size, n_dimensions // batch_size, n_timesteps, 1)

        return cls(times=df.index, values=values_4d, metadata={"source": "dataframe"})

    @classmethod
    def from_numpy(
        cls,
        values: np.ndarray,
        times: pd.DatetimeIndex | pd.RangeIndex | pd.Index | None = None,
        batch_size: int = 1,
    ) -> "TimeSeries":
        """Create a TimeSeries from a numpy array.

        Args:
            values: Numpy array of shape (timesteps,) or (timesteps, dimensions) or (batch, dimension, timestep, value)
            times: Time index (default: RangeIndex)
            batch_size: Number of batches if values need reshaping (default: 1)

        Returns:
            TimeSeries instance

        """
        # Handle different input shapes
        ndim_2d = 2
        ndim_3d = 3
        if values.ndim == 1:
            # (timesteps,) -> (1, 1, timesteps, 1)
            values = values.reshape(1, 1, -1, 1)
        elif values.ndim == ndim_2d:
            # (timesteps, dimensions) -> (batch_size, dimensions//batch_size, timesteps, 1)
            n_timesteps, n_dimensions = values.shape
            values = values.T.reshape(batch_size, n_dimensions // batch_size, n_timesteps, 1)
        elif values.ndim == ndim_3d:
            # (batch, dimension, timestep) -> (batch, dimension, timestep, 1)
            values = values.reshape(*values.shape, 1)
        elif values.ndim != _TIMESERIES_NDIM:
            msg = f"Values must be 1D, 2D, 3D, or 4D array, got {values.ndim}D"
            raise ValueError(msg)

        n_timesteps = values.shape[2]
        if times is None:
            times = pd.RangeIndex(0, n_timesteps)

        return cls(times=times, values=values, metadata={"source": "numpy"})

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        batch_size: int = 1,
    ) -> "TimeSeries":
        """Create a univariate TimeSeries from a pandas Series.

        Args:
            series: Pandas Series with time index
            batch_size: Number of batches (default: 1)

        Returns:
            TimeSeries instance (univariate)

        Raises:
            ValueError: If Series index is not temporal

        """
        if not isinstance(series.index, (pd.DatetimeIndex, pd.RangeIndex)):
            msg = "Series must have DatetimeIndex or RangeIndex"
            raise TypeError(msg)

        # Convert to (batch, 1, timesteps, 1)
        values = series.to_numpy().reshape(batch_size, 1, len(series) // batch_size, 1)
        return cls(times=series.index[:len(series) // batch_size], values=values, metadata={"source": "series"})

    # Conversion methods: TO common formats

    def to_dataframe(self, batch_idx: int = 0) -> pd.DataFrame:
        """Convert TimeSeries to pandas DataFrame.

        Args:
            batch_idx: Index of the batch to convert (default: 0)

        Returns:
            DataFrame with time index and dimension columns

        Raises:
            IndexError: If batch_idx is out of range

        """
        if batch_idx >= self.n_batches:
            msg = f"batch_idx {batch_idx} out of range (n_batches={self.n_batches})"
            raise IndexError(msg)

        # Extract batch: (dimension, timestep, value) -> (timestep, dimension)
        batch_values = self._values[batch_idx, :, :, 0].T

        # Create column names
        columns = [f"dim_{i}" for i in range(self.n_dimensions)]

        return pd.DataFrame(batch_values, index=self._times, columns=columns)

    def to_numpy(self) -> np.ndarray:
        """Extract numpy array from TimeSeries.

        Returns:
            4D numpy array with shape (batch, dimension, timestep, value)

        """
        return self._values

    def to_series(self, batch_idx: int = 0, dim_idx: int = 0) -> pd.Series:
        """Convert TimeSeries to pandas Series (for univariate or single dimension).

        Args:
            batch_idx: Index of the batch to convert (default: 0)
            dim_idx: Index of the dimension to extract (default: 0)

        Returns:
            Series with time index

        Raises:
            IndexError: If batch_idx or dim_idx is out of range

        """
        if batch_idx >= self.n_batches:
            msg = f"batch_idx {batch_idx} out of range (n_batches={self.n_batches})"
            raise IndexError(msg)
        if dim_idx >= self.n_dimensions:
            msg = f"dim_idx {dim_idx} out of range (n_dimensions={self.n_dimensions})"
            raise IndexError(msg)

        # Extract single dimension: (timestep, value) -> (timestep,)
        values = self._values[batch_idx, dim_idx, :, 0]
        return pd.Series(values, index=self._times, name=f"batch_{batch_idx}_dim_{dim_idx}")

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return self.n_timesteps

    def __repr__(self) -> str:
        """Return a string representation of the TimeSeries."""
        time_range = f"times={self._times[0]} to {self._times[-1]}" if len(self._times) > 0 else "times=empty"
        return (
            f"TimeSeries(shape={self.shape}, "
            f"{'univariate' if self.is_univariate else 'multivariate'}, "
            f"{time_range})"
        )


class TabularData(Data):
    """Class representing tabular data structured as (batch, feature).

    This class provides a structured way to handle tabular/non-temporal data
    using numpy arrays with strict type definitions via jaxtyping.

    The data is stored as a 2D numpy array with dimensions:
    - batch: Number of independent samples/rows
    - feature: Number of features/columns
    """

    def __init__(
        self,
        values: Float[np.ndarray, "batch feature"],
        feature_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create 'TabularData' from a numpy array of 'values'.

        Args:
            values: 2D numpy array structured as (batch, feature)
            feature_names: Optional list of feature names
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If values is not a 2D array
            ValueError: If feature_names length doesn't match feature dimension

        """
        if values.ndim != _TABULAR_NDIM:
            msg = f"Values must be a {_TABULAR_NDIM}D array (batch, feature), got {values.ndim}D array"
            raise ValueError(msg)

        if feature_names is not None and len(feature_names) != values.shape[1]:
            msg = f"Feature names length ({len(feature_names)}) must match feature dimension ({values.shape[1]})"
            raise ValueError(msg)

        self._values = values
        self._feature_names = feature_names
        self._metadata = metadata or {}

    @property
    def values(self) -> Float[np.ndarray, "batch feature"]:
        """Return the values array.

        Returns:
            2D numpy array with shape (batch, feature).

        """
        return self._values

    @property
    def feature_names(self) -> list[str] | None:
        """Return the feature names."""
        return self._feature_names

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata dictionary."""
        return self._metadata

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the values array as (batch, feature)."""
        return self._values.shape

    @property
    def n_samples(self) -> int:
        """Return the number of samples/rows."""
        return self._values.shape[0]

    @property
    def n_features(self) -> int:
        """Return the number of features/columns."""
        return self._values.shape[1]

    # Conversion methods: FROM common formats

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TabularData":
        """Create TabularData from a pandas DataFrame.

        Args:
            df: DataFrame with samples as rows and features as columns

        Returns:
            TabularData instance

        """
        values = df.to_numpy()
        feature_names = list(df.columns)
        return cls(values=values, feature_names=feature_names, metadata={"source": "dataframe"})

    @classmethod
    def from_numpy(
        cls,
        values: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "TabularData":
        """Create TabularData from a numpy array.

        Args:
            values: 2D numpy array with shape (samples, features)
            feature_names: Optional list of feature names

        Returns:
            TabularData instance

        Raises:
            ValueError: If values is not a 2D array

        """
        if values.ndim == 1:
            # Convert 1D array to 2D with single feature
            values = values.reshape(-1, 1)
        elif values.ndim != _TABULAR_NDIM:
            msg = f"Values must be 1D or 2D array, got {values.ndim}D"
            raise ValueError(msg)

        return cls(values=values, feature_names=feature_names, metadata={"source": "numpy"})

    @classmethod
    def from_dict(
        cls,
        data: dict[str, list | np.ndarray],
    ) -> "TabularData":
        """Create TabularData from a dictionary.

        Args:
            data: Dictionary with feature names as keys and lists/arrays as values

        Returns:
            TabularData instance

        Raises:
            ValueError: If arrays have different lengths

        """
        if not data:
            msg = "Dictionary cannot be empty"
            raise ValueError(msg)

        # Check all arrays have the same length
        lengths = {key: len(value) for key, value in data.items()}
        if len(set(lengths.values())) > 1:
            msg = f"All arrays must have the same length, got {lengths}"
            raise ValueError(msg)

        # Convert to numpy array
        feature_names = list(data.keys())
        values = np.column_stack([np.asarray(data[key]) for key in feature_names])

        return cls(values=values, feature_names=feature_names, metadata={"source": "dict"})

    # Conversion methods: TO common formats

    def to_dataframe(self) -> pd.DataFrame:
        """Convert TabularData to pandas DataFrame.

        Returns:
            DataFrame with feature names as columns

        """
        columns = self._feature_names if self._feature_names else [f"feature_{i}" for i in range(self.n_features)]
        return pd.DataFrame(self._values, columns=columns)

    def to_numpy(self) -> np.ndarray:
        """Extract numpy array from TabularData.

        Returns:
            2D numpy array with shape (batch, feature)

        """
        return self._values

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert TabularData to dictionary format.

        Returns:
            Dictionary with feature names as keys and column arrays as values

        """
        columns = self._feature_names if self._feature_names else [f"feature_{i}" for i in range(self.n_features)]
        return {col: self._values[:, i] for i, col in enumerate(columns)}

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __repr__(self) -> str:
        """Return a string representation of the TabularData."""
        feature_info = f", features={self._feature_names}" if self._feature_names else ""
        return f"TabularData(shape={self.shape}{feature_info})"
