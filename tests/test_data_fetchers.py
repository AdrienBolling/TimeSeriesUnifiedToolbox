"""Tests for CSV and JSON Data Fetcher Nodes."""

import tempfile
from pathlib import Path

import pytest

from tsut.core.common.data.types import TabularData, TimeSeries
from tsut.core.nodes.source.csv_fetcher import CSVDataFetcher, CSVDataFetcherConfig
from tsut.core.nodes.source.json_fetcher import JSONDataFetcher, JSONDataFetcherConfig


class TestCSVDataFetcher:
    """Tests for the CSVDataFetcher class."""

    @pytest.fixture
    def temp_csv_timeseries(self):
        """Create a temporary CSV file with time series data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,value1,value2\n")
            f.write("2020-01-01,1.0,2.0\n")
            f.write("2020-01-02,3.0,4.0\n")
            f.write("2020-01-03,5.0,6.0\n")
            f.write("2020-01-04,7.0,8.0\n")
            f.write("2020-01-05,9.0,10.0\n")
            return Path(f.name)

    @pytest.fixture
    def temp_csv_tabular(self):
        """Create a temporary CSV file with tabular data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("feature1,feature2,feature3\n")
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            f.write("7.0,8.0,9.0\n")
            return Path(f.name)

    def test_csv_fetcher_timeseries_basic(self, temp_csv_timeseries: Path) -> None:
        """Test basic CSV fetcher with time series data."""
        config = CSVDataFetcherConfig(
            file_path=str(temp_csv_timeseries),
            data_type="timeseries",
            time_column="timestamp",
            value_columns=["value1", "value2"],
            batch_size=1,
        )

        fetcher = CSVDataFetcher(config=config)
        result = fetcher.node_transform({})

        assert "output" in result
        assert isinstance(result["output"], TimeSeries)

        ts = result["output"]
        assert ts.n_timesteps == 5
        assert ts.n_dimensions == 2
        assert ts.n_batches == 1

        # Cleanup
        temp_csv_timeseries.unlink()

    def test_csv_fetcher_tabular_basic(self, temp_csv_tabular: Path) -> None:
        """Test basic CSV fetcher with tabular data."""
        config = CSVDataFetcherConfig(
            file_path=str(temp_csv_tabular),
            data_type="tabular",
        )

        fetcher = CSVDataFetcher(config=config)
        result = fetcher.node_transform({})

        assert "output" in result
        assert isinstance(result["output"], TabularData)

        td = result["output"]
        assert td.n_samples == 3
        assert td.n_features == 3

        # Cleanup
        temp_csv_tabular.unlink()

    def test_csv_fetcher_file_not_found(self):
        """Test CSV fetcher with non-existent file."""
        config = CSVDataFetcherConfig(
            file_path="nonexistent.csv",
            data_type="tabular",
        )

        fetcher = CSVDataFetcher(config=config)

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            fetcher.fetch_data()

    def test_csv_fetcher_missing_time_column(self, temp_csv_timeseries: Path) -> None:
        """Test CSV fetcher with missing time column specification."""
        config = CSVDataFetcherConfig(
            file_path=str(temp_csv_timeseries),
            data_type="timeseries",
            time_column=None,
        )

        fetcher = CSVDataFetcher(config=config)

        with pytest.raises(ValueError, match="time_column must be specified"):
            fetcher.fetch_data()

        # Cleanup
        temp_csv_timeseries.unlink()

    def test_csv_fetcher_invalid_time_column(self, temp_csv_timeseries: Path) -> None:
        """Test CSV fetcher with invalid time column name."""
        config = CSVDataFetcherConfig(
            file_path=str(temp_csv_timeseries),
            data_type="timeseries",
            time_column="invalid_column",
        )

        fetcher = CSVDataFetcher(config=config)

        with pytest.raises(ValueError, match="Time column 'invalid_column' not found"):
            fetcher.fetch_data()

        # Cleanup
        temp_csv_timeseries.unlink()

    def test_csv_fetcher_value_columns_selection(self, temp_csv_timeseries: Path) -> None:
        """Test CSV fetcher with specific value columns."""
        config = CSVDataFetcherConfig(
            file_path=str(temp_csv_timeseries),
            data_type="timeseries",
            time_column="timestamp",
            value_columns=["value1"],
            batch_size=1,
        )

        fetcher = CSVDataFetcher(config=config)
        result = fetcher.fetch_data()

        assert isinstance(result, TimeSeries)
        assert result.n_dimensions == 1

        # Cleanup
        temp_csv_timeseries.unlink()

    def test_csv_fetcher_missing_value_columns(self, temp_csv_timeseries: Path) -> None:
        """Test CSV fetcher with missing value columns."""
        config = CSVDataFetcherConfig(
            file_path=str(temp_csv_timeseries),
            data_type="timeseries",
            time_column="timestamp",
            value_columns=["nonexistent"],
        )

        fetcher = CSVDataFetcher(config=config)

        with pytest.raises(ValueError, match="Value columns .* not found"):
            fetcher.fetch_data()

        # Cleanup
        temp_csv_timeseries.unlink()

    def test_csv_fetcher_empty_file(self):
        """Test CSV fetcher with empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n")
            temp_file = Path(f.name)

        config = CSVDataFetcherConfig(
            file_path=str(temp_file),
            data_type="tabular",
        )

        fetcher = CSVDataFetcher(config=config)

        with pytest.raises(ValueError, match="CSV file is empty"):
            fetcher.fetch_data()

        # Cleanup
        temp_file.unlink()

    def test_csv_fetcher_output_ports(self):
        """Test that output ports are correctly configured."""
        config_ts = CSVDataFetcherConfig(
            file_path="dummy.csv",
            data_type="timeseries",
            time_column="time",
        )
        assert "output" in config_ts.out_ports
        assert config_ts.out_ports["output"].type == TimeSeries

        config_tabular = CSVDataFetcherConfig(
            file_path="dummy.csv",
            data_type="tabular",
        )
        assert "output" in config_tabular.out_ports
        assert config_tabular.out_ports["output"].type == TabularData


class TestJSONDataFetcher:
    """Tests for the JSONDataFetcher class."""

    @pytest.fixture
    def temp_json_timeseries_records(self):
        """Create a temporary JSON file with time series data (records format)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("[")
            f.write('{"timestamp": "2020-01-01", "value1": 1.0, "value2": 2.0},')
            f.write('{"timestamp": "2020-01-02", "value1": 3.0, "value2": 4.0},')
            f.write('{"timestamp": "2020-01-03", "value1": 5.0, "value2": 6.0}')
            f.write("]")
            return Path(f.name)

    @pytest.fixture
    def temp_json_tabular_columns(self):
        """Create a temporary JSON file with tabular data (columns format)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"feature1": [1.0, 4.0, 7.0], "feature2": [2.0, 5.0, 8.0], "feature3": [3.0, 6.0, 9.0]}')
            return Path(f.name)

    def test_json_fetcher_timeseries_basic(self, temp_json_timeseries_records: Path) -> None:
        """Test basic JSON fetcher with time series data."""
        config = JSONDataFetcherConfig(
            file_path=str(temp_json_timeseries_records),
            data_type="timeseries",
            time_column="timestamp",
            value_columns=["value1", "value2"],
            batch_size=1,
            orient="records",
        )

        fetcher = JSONDataFetcher(config=config)
        result = fetcher.node_transform({})

        assert "output" in result
        assert isinstance(result["output"], TimeSeries)

        ts = result["output"]
        assert ts.n_timesteps == 3
        assert ts.n_dimensions == 2
        assert ts.n_batches == 1

        # Cleanup
        temp_json_timeseries_records.unlink()

    def test_json_fetcher_tabular_basic(self, temp_json_tabular_columns: Path) -> None:
        """Test basic JSON fetcher with tabular data."""
        config = JSONDataFetcherConfig(
            file_path=str(temp_json_tabular_columns),
            data_type="tabular",
            orient="columns",
        )

        fetcher = JSONDataFetcher(config=config)
        result = fetcher.node_transform({})

        assert "output" in result
        assert isinstance(result["output"], TabularData)

        td = result["output"]
        assert td.n_samples == 3
        assert td.n_features == 3

        # Cleanup
        temp_json_tabular_columns.unlink()

    def test_json_fetcher_file_not_found(self):
        """Test JSON fetcher with non-existent file."""
        config = JSONDataFetcherConfig(
            file_path="nonexistent.json",
            data_type="tabular",
        )

        fetcher = JSONDataFetcher(config=config)

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            fetcher.fetch_data()

    def test_json_fetcher_missing_time_column(self, temp_json_timeseries_records: Path) -> None:
        """Test JSON fetcher with missing time column specification."""
        config = JSONDataFetcherConfig(
            file_path=str(temp_json_timeseries_records),
            data_type="timeseries",
            time_column=None,
            orient="records",
        )

        fetcher = JSONDataFetcher(config=config)

        with pytest.raises(ValueError, match="time_column must be specified"):
            fetcher.fetch_data()

        # Cleanup
        temp_json_timeseries_records.unlink()

    def test_json_fetcher_invalid_time_column(self, temp_json_timeseries_records: Path) -> None:
        """Test JSON fetcher with invalid time column name."""
        config = JSONDataFetcherConfig(
            file_path=str(temp_json_timeseries_records),
            data_type="timeseries",
            time_column="invalid_column",
            orient="records",
        )

        fetcher = JSONDataFetcher(config=config)

        with pytest.raises(ValueError, match="Time column 'invalid_column' not found"):
            fetcher.fetch_data()

        # Cleanup
        temp_json_timeseries_records.unlink()

    def test_json_fetcher_value_columns_selection(self, temp_json_timeseries_records: Path) -> None:
        """Test JSON fetcher with specific value columns."""
        config = JSONDataFetcherConfig(
            file_path=str(temp_json_timeseries_records),
            data_type="timeseries",
            time_column="timestamp",
            value_columns=["value1"],
            batch_size=1,
            orient="records",
        )

        fetcher = JSONDataFetcher(config=config)
        result = fetcher.fetch_data()

        assert isinstance(result, TimeSeries)
        assert result.n_dimensions == 1

        # Cleanup
        temp_json_timeseries_records.unlink()

    def test_json_fetcher_missing_value_columns(self, temp_json_timeseries_records: Path) -> None:
        """Test JSON fetcher with missing value columns."""
        config = JSONDataFetcherConfig(
            file_path=str(temp_json_timeseries_records),
            data_type="timeseries",
            time_column="timestamp",
            value_columns=["nonexistent"],
            orient="records",
        )

        fetcher = JSONDataFetcher(config=config)

        with pytest.raises(ValueError, match="Value columns .* not found"):
            fetcher.fetch_data()

        # Cleanup
        temp_json_timeseries_records.unlink()

    def test_json_fetcher_empty_file(self):
        """Test JSON fetcher with empty JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"col1": [], "col2": []}')
            temp_file = Path(f.name)

        config = JSONDataFetcherConfig(
            file_path=str(temp_file),
            data_type="tabular",
            orient="columns",
        )

        fetcher = JSONDataFetcher(config=config)

        with pytest.raises(ValueError, match="JSON file is empty"):
            fetcher.fetch_data()

        # Cleanup
        temp_file.unlink()

    def test_json_fetcher_output_ports(self):
        """Test that output ports are correctly configured."""
        config_ts = JSONDataFetcherConfig(
            file_path="dummy.json",
            data_type="timeseries",
            time_column="time",
        )
        assert "output" in config_ts.out_ports
        assert config_ts.out_ports["output"].type == TimeSeries

        config_tabular = JSONDataFetcherConfig(
            file_path="dummy.json",
            data_type="tabular",
        )
        assert "output" in config_tabular.out_ports
        assert config_tabular.out_ports["output"].type == TabularData

    def test_json_fetcher_invalid_json(self):
        """Test JSON fetcher with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json syntax')
            temp_file = Path(f.name)

        config = JSONDataFetcherConfig(
            file_path=str(temp_file),
            data_type="tabular",
        )

        fetcher = JSONDataFetcher(config=config)

        with pytest.raises(ValueError, match="Failed to read JSON file"):
            fetcher.fetch_data()

        # Cleanup
        temp_file.unlink()
