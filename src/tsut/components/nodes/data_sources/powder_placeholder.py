"""Placeholder for RTP powder project data source node."""

from tsut.core.nodes.data_source.data_source import DataSourceConfig, DataSourceMetadata, DataSourceNode, DataSourceRunningConfig
from tsut.core.nodes.node import Port

import pandas as pd


class PowderDataSourceMetadata(DataSourceMetadata):
    """Metadata for the RTP powder project data source node."""
    
    node_name: str = "RTP Powder Data Source"
    input_type: str = "None"  # Data sources have no input
    output_type: str = "pd.DataFrame" # Output is a dictionary
    node_description: str = "Data source node for the RTP powder project, responsible for fetching and preparing the data for the project."


class PowderDataSourceRunningConfig(DataSourceRunningConfig):
    """Configuration for running the RTP powder project data source node."""
    # Add any specific fields needed for running the RTP powder data source if necessary

class PowderDataSourceConfig(DataSourceConfig):
    """Configuration for the RTP powder project data source node."""

    running_config: PowderDataSourceRunningConfig = PowderDataSourceRunningConfig()
    in_ports: dict[str, Port] = {}
    out_ports: dict[str, Port] = {"prepared_data": Port(type=pd.DataFrame, desc="Prepared data for the RTP powder project", mode=["training", "evaluation"])}

class PowderDataSourceNode(DataSourceNode[dict[str, pd.DataFrame]]):
    """Data source node for the RTP powder project, responsible for fetching and preparing the data for the project."""

    metadata = PowderDataSourceMetadata()

    def setup_source(self) -> None:
        """Set up the data source (e.g., establish connections, load resources)."""
        # TODO @ Soumi : implement the setup for the RTP powder data source node here, e.g., establish connections, load resources, etc.
        raise NotImplementedError("Setup for the RTP powder data source node is not implemented yet.")

    def fetch_data(self) -> dict[str, pd.DataFrame]:
        """Fetch data from the source.

        Returns:
            Fetched data as a dictionary of pandas DataFrames.

        """
        # TODO @ Soumi : implement the data getting and engineering here then return it
        # typically return something that respects the ports naming convention defined above, like : return {"prepared_data": prepared_df}
        raise NotImplementedError("Data fetching for the RTP powder data source node is not implemented yet.")