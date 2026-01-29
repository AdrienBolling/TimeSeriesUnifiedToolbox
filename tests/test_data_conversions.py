"""Tests for data conversion functions."""

import numpy as np
import pandas as pd
import pytest

from tsut.core.common.data.types import TabularData, TimeSeries


class TestTimeSeriesConversions:
    """Tests for TimeSeries conversion methods."""

    def test_from_dataframe_univariate(self):
        """Test creating TimeSeries from DataFrame (univariate)."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame({"value": np.arange(10)}, index=times)

        ts = TimeSeries.from_dataframe(df, batch_size=1)

        assert ts.shape == (1, 1, 10, 1)
        assert ts.is_univariate
        assert isinstance(ts.times, pd.DatetimeIndex)

    def test_from_dataframe_multivariate(self):
        """Test creating TimeSeries from DataFrame (multivariate)."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame({
            "sensor1": np.arange(10),
            "sensor2": np.arange(10, 20),
            "sensor3": np.arange(20, 30)
        }, index=times)

        ts = TimeSeries.from_dataframe(df, batch_size=1)

        assert ts.shape == (1, 3, 10, 1)
        assert ts.is_multivariate
        assert ts.n_dimensions == 3

    def test_from_dataframe_with_rangeindex(self):
        """Test creating TimeSeries from DataFrame with RangeIndex."""
        df = pd.DataFrame({"value": np.arange(10)}, index=pd.RangeIndex(0, 10))

        ts = TimeSeries.from_dataframe(df, batch_size=1)

        assert isinstance(ts.times, pd.RangeIndex)
        assert len(ts) == 10

    def test_from_dataframe_default_index(self):
        """Test creating TimeSeries from DataFrame with default (Range) index."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        # Default index is RangeIndex, which should work
        ts = TimeSeries.from_dataframe(df)

        assert isinstance(ts.times, pd.RangeIndex)
        assert ts.shape == (1, 1, 3, 1)

    def test_from_numpy_1d(self):
        """Test creating TimeSeries from 1D numpy array."""
        values = np.arange(10)

        ts = TimeSeries.from_numpy(values)

        assert ts.shape == (1, 1, 10, 1)
        assert ts.is_univariate

    def test_from_numpy_2d(self):
        """Test creating TimeSeries from 2D numpy array."""
        values = np.random.randn(10, 3)  # (timesteps, dimensions)

        ts = TimeSeries.from_numpy(values, batch_size=1)

        assert ts.shape == (1, 3, 10, 1)
        assert ts.n_dimensions == 3

    def test_from_numpy_3d(self):
        """Test creating TimeSeries from 3D numpy array."""
        values = np.random.randn(2, 3, 10)  # (batch, dimension, timestep)

        ts = TimeSeries.from_numpy(values)

        assert ts.shape == (2, 3, 10, 1)

    def test_from_numpy_4d(self):
        """Test creating TimeSeries from 4D numpy array."""
        values = np.random.randn(2, 3, 10, 1)

        ts = TimeSeries.from_numpy(values)

        assert ts.shape == (2, 3, 10, 1)

    def test_from_numpy_with_times(self):
        """Test creating TimeSeries with custom time index."""
        values = np.arange(10)
        times = pd.date_range("2024-01-01", periods=10, freq="D")

        ts = TimeSeries.from_numpy(values, times=times)

        assert isinstance(ts.times, pd.DatetimeIndex)
        assert len(ts.times) == 10

    def test_from_series(self):
        """Test creating TimeSeries from pandas Series."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        series = pd.Series(np.arange(10), index=times)

        ts = TimeSeries.from_series(series)

        assert ts.shape == (1, 1, 10, 1)
        assert ts.is_univariate

    def test_from_series_default_index(self):
        """Test creating TimeSeries from Series with default (Range) index."""
        series = pd.Series([1, 2, 3])

        # Default index is RangeIndex, which should work
        ts = TimeSeries.from_series(series)

        assert isinstance(ts.times, pd.RangeIndex)
        assert ts.shape == (1, 1, 3, 1)

    def test_to_dataframe(self):
        """Test converting TimeSeries to DataFrame."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        values = np.random.randn(2, 3, 10, 1)
        ts = TimeSeries(times=times, values=values)

        df = ts.to_dataframe(batch_idx=0)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 3)
        assert list(df.columns) == ["dim_0", "dim_1", "dim_2"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_to_dataframe_invalid_batch(self):
        """Test that invalid batch index raises error."""
        times = pd.RangeIndex(0, 10)
        values = np.random.randn(2, 3, 10, 1)
        ts = TimeSeries(times=times, values=values)

        with pytest.raises(IndexError, match="batch_idx.*out of range"):
            ts.to_dataframe(batch_idx=5)

    def test_to_numpy(self):
        """Test extracting numpy array from TimeSeries."""
        times = pd.RangeIndex(0, 10)
        values = np.random.randn(2, 3, 10, 1)
        ts = TimeSeries(times=times, values=values)

        extracted = ts.to_numpy()

        assert np.array_equal(extracted, values)
        assert extracted.shape == (2, 3, 10, 1)

    def test_to_series_univariate(self):
        """Test converting univariate TimeSeries to Series."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        values = np.arange(10).reshape(1, 1, 10, 1)
        ts = TimeSeries(times=times, values=values)

        series = ts.to_series()

        assert isinstance(series, pd.Series)
        assert len(series) == 10
        assert isinstance(series.index, pd.DatetimeIndex)

    def test_to_series_multivariate(self):
        """Test converting single dimension from multivariate TimeSeries."""
        times = pd.RangeIndex(0, 10)
        values = np.random.randn(2, 3, 10, 1)
        ts = TimeSeries(times=times, values=values)

        series = ts.to_series(batch_idx=0, dim_idx=1)

        assert isinstance(series, pd.Series)
        assert len(series) == 10
        assert series.name == "batch_0_dim_1"

    def test_to_series_invalid_indices(self):
        """Test that invalid indices raise errors."""
        times = pd.RangeIndex(0, 10)
        values = np.random.randn(2, 3, 10, 1)
        ts = TimeSeries(times=times, values=values)

        with pytest.raises(IndexError, match="batch_idx.*out of range"):
            ts.to_series(batch_idx=5)

        with pytest.raises(IndexError, match="dim_idx.*out of range"):
            ts.to_series(dim_idx=5)

    def test_roundtrip_dataframe(self):
        """Test roundtrip conversion: DataFrame -> TimeSeries -> DataFrame."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        original_df = pd.DataFrame({
            "col1": np.arange(10),
            "col2": np.arange(10, 20)
        }, index=times)

        ts = TimeSeries.from_dataframe(original_df)
        result_df = ts.to_dataframe()

        assert result_df.shape == original_df.shape
        assert len(result_df) == len(original_df)


class TestTabularDataConversions:
    """Tests for TabularData conversion methods."""

    def test_from_dataframe(self):
        """Test creating TabularData from DataFrame."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "feature3": [100, 200, 300, 400, 500]
        })

        td = TabularData.from_dataframe(df)

        assert td.shape == (5, 3)
        assert td.feature_names == ["feature1", "feature2", "feature3"]
        assert td.metadata["source"] == "dataframe"

    def test_from_numpy_2d(self):
        """Test creating TabularData from 2D numpy array."""
        values = np.random.randn(10, 5)

        td = TabularData.from_numpy(values)

        assert td.shape == (10, 5)
        assert td.feature_names is None

    def test_from_numpy_2d_with_names(self):
        """Test creating TabularData from 2D numpy array with feature names."""
        values = np.random.randn(10, 3)
        feature_names = ["a", "b", "c"]

        td = TabularData.from_numpy(values, feature_names=feature_names)

        assert td.feature_names == feature_names

    def test_from_numpy_1d(self):
        """Test creating TabularData from 1D numpy array."""
        values = np.arange(10)

        td = TabularData.from_numpy(values)

        assert td.shape == (10, 1)

    def test_from_numpy_invalid_dims(self):
        """Test that 3D array raises error."""
        values = np.random.randn(5, 3, 2)

        with pytest.raises(ValueError, match="1D or 2D array"):
            TabularData.from_numpy(values)

    def test_from_dict(self):
        """Test creating TabularData from dictionary."""
        data = {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "feature3": [100, 200, 300, 400, 500]
        }

        td = TabularData.from_dict(data)

        assert td.shape == (5, 3)
        assert td.feature_names == ["feature1", "feature2", "feature3"]

    def test_from_dict_numpy_arrays(self):
        """Test creating TabularData from dict with numpy arrays."""
        data = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6])
        }

        td = TabularData.from_dict(data)

        assert td.shape == (3, 2)

    def test_from_dict_empty(self):
        """Test that empty dict raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TabularData.from_dict({})

    def test_from_dict_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        data = {
            "a": [1, 2, 3],
            "b": [4, 5]
        }

        with pytest.raises(ValueError, match="same length"):
            TabularData.from_dict(data)

    def test_to_dataframe(self):
        """Test converting TabularData to DataFrame."""
        values = np.random.randn(10, 3)
        feature_names = ["a", "b", "c"]
        td = TabularData(values=values, feature_names=feature_names)

        df = td.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 3)
        assert list(df.columns) == feature_names

    def test_to_dataframe_no_names(self):
        """Test converting TabularData without feature names."""
        values = np.random.randn(10, 3)
        td = TabularData(values=values)

        df = td.to_dataframe()

        assert list(df.columns) == ["feature_0", "feature_1", "feature_2"]

    def test_to_numpy(self):
        """Test extracting numpy array from TabularData."""
        values = np.random.randn(10, 3)
        td = TabularData(values=values)

        extracted = td.to_numpy()

        assert np.array_equal(extracted, values)
        assert extracted.shape == (10, 3)

    def test_to_dict(self):
        """Test converting TabularData to dictionary."""
        values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        feature_names = ["a", "b", "c"]
        td = TabularData(values=values, feature_names=feature_names)

        result = td.to_dict()

        assert list(result.keys()) == feature_names
        assert np.array_equal(result["a"], np.array([1, 4, 7]))
        assert np.array_equal(result["b"], np.array([2, 5, 8]))
        assert np.array_equal(result["c"], np.array([3, 6, 9]))

    def test_to_dict_no_names(self):
        """Test converting TabularData to dict without feature names."""
        values = np.random.randn(10, 3)
        td = TabularData(values=values)

        result = td.to_dict()

        assert list(result.keys()) == ["feature_0", "feature_1", "feature_2"]

    def test_roundtrip_dataframe(self):
        """Test roundtrip conversion: DataFrame -> TabularData -> DataFrame."""
        original_df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9]
        })

        td = TabularData.from_dataframe(original_df)
        result_df = td.to_dataframe()

        assert result_df.shape == original_df.shape
        assert list(result_df.columns) == list(original_df.columns)
        assert np.allclose(result_df.values, original_df.values)

    def test_roundtrip_dict(self):
        """Test roundtrip conversion: dict -> TabularData -> dict."""
        original_dict = {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        }

        td = TabularData.from_dict(original_dict)
        result_dict = td.to_dict()

        assert list(result_dict.keys()) == list(original_dict.keys())
        for key, value in original_dict.items():
            assert np.allclose(result_dict[key], value)
