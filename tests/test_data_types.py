"""Tests for TimeSeries and TabularData classes."""

import numpy as np
import pandas as pd
import pytest

from tsut.core.common.data.types import TabularData, TimeSeries


class TestTimeSeries:
    """Tests for the TimeSeries class."""

    def test_create_univariate_timeseries(self):
        """Test creating a univariate time series."""
        times = pd.RangeIndex(start=0, stop=10, step=1)
        values = np.random.randn(2, 1, 10, 1)  # (batch, dimension, timestep, value)

        ts = TimeSeries(times=times, values=values)

        assert ts.shape == (2, 1, 10, 1)
        assert ts.n_batches == 2
        assert ts.n_dimensions == 1
        assert ts.n_timesteps == 10
        assert ts.n_values == 1
        assert ts.is_univariate
        assert not ts.is_multivariate
        assert len(ts) == 10

    def test_create_multivariate_timeseries(self):
        """Test creating a multivariate time series."""
        times = pd.RangeIndex(start=0, stop=20, step=1)
        values = np.random.randn(3, 5, 20, 1)  # (batch, dimension, timestep, value)

        ts = TimeSeries(times=times, values=values)

        assert ts.shape == (3, 5, 20, 1)
        assert ts.n_batches == 3
        assert ts.n_dimensions == 5
        assert ts.n_timesteps == 20
        assert ts.n_values == 1
        assert not ts.is_univariate
        assert ts.is_multivariate

    def test_create_timeseries_with_datetime_index(self):
        """Test creating a time series with DatetimeIndex."""
        times = pd.date_range(start="2020-01-01", periods=15, freq="D")
        values = np.random.randn(1, 2, 15, 1)

        ts = TimeSeries(times=times, values=values)

        assert isinstance(ts.times, pd.DatetimeIndex)
        assert len(ts.times) == 15
        assert ts.n_timesteps == 15

    def test_create_timeseries_with_metadata(self):
        """Test creating a time series with metadata."""
        times = pd.RangeIndex(start=0, stop=5, step=1)
        values = np.random.randn(1, 1, 5, 1)
        metadata = {"source": "test", "unit": "meters"}

        ts = TimeSeries(times=times, values=values, metadata=metadata)

        assert ts.metadata == metadata
        assert ts.metadata["source"] == "test"
        assert ts.metadata["unit"] == "meters"

    def test_timeseries_wrong_dimensions_raises_error(self):
        """Test that creating a time series with wrong dimensions raises ValueError."""
        times = pd.RangeIndex(start=0, stop=10, step=1)

        # 3D array instead of 4D
        values_3d = np.random.randn(2, 1, 10)
        with pytest.raises(ValueError, match="Values must be a 4D array"):
            TimeSeries(times=times, values=values_3d)

        # 2D array
        values_2d = np.random.randn(2, 10)
        with pytest.raises(ValueError, match="Values must be a 4D array"):
            TimeSeries(times=times, values=values_2d)

    def test_timeseries_mismatched_time_length_raises_error(self):
        """Test that mismatched time index length raises ValueError."""
        times = pd.RangeIndex(start=0, stop=5, step=1)  # 5 timesteps
        values = np.random.randn(1, 1, 10, 1)  # 10 timesteps

        with pytest.raises(ValueError, match="Time index length.*must match timestep dimension"):
            TimeSeries(times=times, values=values)

    def test_timeseries_properties(self):
        """Test TimeSeries properties."""
        times = pd.RangeIndex(start=0, stop=8, step=1)
        values = np.ones((4, 3, 8, 2))

        ts = TimeSeries(times=times, values=values)

        assert np.array_equal(ts.values, values)
        assert ts.times.equals(times)
        assert isinstance(ts.metadata, dict)

    def test_timeseries_repr(self):
        """Test TimeSeries string representation."""
        times = pd.RangeIndex(start=0, stop=5, step=1)
        values = np.random.randn(1, 1, 5, 1)

        ts = TimeSeries(times=times, values=values)
        repr_str = repr(ts)

        assert "TimeSeries" in repr_str
        assert "shape=(1, 1, 5, 1)" in repr_str
        assert "univariate" in repr_str


class TestTabularData:
    """Tests for the TabularData class."""

    def test_create_tabular_data(self):
        """Test creating tabular data."""
        values = np.random.randn(100, 5)  # (batch, feature)

        td = TabularData(values=values)

        assert td.shape == (100, 5)
        assert td.n_samples == 100
        assert td.n_features == 5
        assert len(td) == 100

    def test_create_tabular_data_with_feature_names(self):
        """Test creating tabular data with feature names."""
        values = np.random.randn(50, 3)
        feature_names = ["feature1", "feature2", "feature3"]

        td = TabularData(values=values, feature_names=feature_names)

        assert td.feature_names == feature_names
        assert td.n_features == 3

    def test_create_tabular_data_with_metadata(self):
        """Test creating tabular data with metadata."""
        values = np.random.randn(20, 4)
        metadata = {"dataset": "iris", "version": "1.0"}

        td = TabularData(values=values, metadata=metadata)

        assert td.metadata == metadata
        assert td.metadata["dataset"] == "iris"

    def test_tabular_data_wrong_dimensions_raises_error(self):
        """Test that creating tabular data with wrong dimensions raises ValueError."""
        # 3D array instead of 2D
        values_3d = np.random.randn(10, 5, 2)
        with pytest.raises(ValueError, match="Values must be a 2D array"):
            TabularData(values=values_3d)

        # 1D array
        values_1d = np.random.randn(10)
        with pytest.raises(ValueError, match="Values must be a 2D array"):
            TabularData(values=values_1d)

    def test_tabular_data_mismatched_feature_names_raises_error(self):
        """Test that mismatched feature names length raises ValueError."""
        values = np.random.randn(10, 5)  # 5 features
        feature_names = ["f1", "f2", "f3"]  # Only 3 names

        with pytest.raises(ValueError, match="Feature names length.*must match feature dimension"):
            TabularData(values=values, feature_names=feature_names)

    def test_tabular_data_properties(self):
        """Test TabularData properties."""
        values = np.ones((30, 4))
        feature_names = ["a", "b", "c", "d"]

        td = TabularData(values=values, feature_names=feature_names)

        assert np.array_equal(td.values, values)
        assert td.feature_names == feature_names
        assert isinstance(td.metadata, dict)

    def test_tabular_data_repr(self):
        """Test TabularData string representation."""
        values = np.random.randn(10, 3)
        feature_names = ["x", "y", "z"]

        td = TabularData(values=values, feature_names=feature_names)
        repr_str = repr(td)

        assert "TabularData" in repr_str
        assert "shape=(10, 3)" in repr_str
        assert "features=['x', 'y', 'z']" in repr_str

    def test_tabular_data_repr_without_feature_names(self):
        """Test TabularData string representation without feature names."""
        values = np.random.randn(15, 2)

        td = TabularData(values=values)
        repr_str = repr(td)

        assert "TabularData" in repr_str
        assert "shape=(15, 2)" in repr_str
        assert "features=" not in repr_str
