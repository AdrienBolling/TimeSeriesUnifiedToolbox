"""Tests for the :class:`TabularCSVFetcher` data-source node."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tsut.components.nodes.data_sources.tabular_csv_fetcher import (
    TabularCSVFetcher,
    TabularCSVFetcherConfig,
    TabularCSVFetcherRunningConfig,
)


class TestTabularCSVFetcherLifecycle:
    def test_setup_and_fetch_returns_dataframe(
        self, csv_and_context_files, numerical_dataframe
    ) -> None:
        csv_path, ctx_path = csv_and_context_files

        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(csv_path),
                context_path=str(ctx_path),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        node.setup_source()

        out = node.fetch_data()
        df, ctx = out["output"]
        pd.testing.assert_frame_equal(df, numerical_dataframe)
        assert ctx.columns == list(numerical_dataframe.columns)

    def test_missing_csv_raises(self, tmp_path) -> None:
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(tmp_path / "does_not_exist.csv"),
                context_path=str(tmp_path / "does_not_exist.json"),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(FileNotFoundError, match="CSV"):
            node.setup_source()

    def test_missing_context_raises(self, tmp_path, numerical_dataframe) -> None:
        csv_path = tmp_path / "data.csv"
        numerical_dataframe.to_csv(csv_path, index=False)
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(csv_path),
                context_path=str(tmp_path / "missing.json"),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(FileNotFoundError, match="Context"):
            node.setup_source()

    def test_context_length_mismatch_raises(
        self, tmp_path, numerical_dataframe
    ) -> None:
        csv_path = tmp_path / "data.csv"
        ctx_path = tmp_path / "ctx.json"
        numerical_dataframe.to_csv(csv_path, index=False)
        # Context with fewer columns than the CSV.
        ctx_path.write_text(
            json.dumps(
                {
                    "columns": ["f0"],
                    "dtypes": ["float64"],
                    "categories": ["numerical_data"],
                }
            )
        )
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(csv_path),
                context_path=str(ctx_path),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(ValueError, match="does not match"):
            node.setup_source()

    def test_fetch_before_setup_raises(self, tmp_path) -> None:
        cfg = TabularCSVFetcherConfig(
            running_config=TabularCSVFetcherRunningConfig(
                csv_path=str(tmp_path / "noop.csv"),
                context_path=str(tmp_path / "noop.json"),
            ),
        )
        node = TabularCSVFetcher(config=cfg)
        with pytest.raises(RuntimeError, match="Data not loaded"):
            node.fetch_data()
