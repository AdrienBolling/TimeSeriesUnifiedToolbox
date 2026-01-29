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

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __repr__(self) -> str:
        """Return a string representation of the TabularData."""
        feature_info = f", features={self._feature_names}" if self._feature_names else ""
        return f"TabularData(shape={self.shape}{feature_info})"
