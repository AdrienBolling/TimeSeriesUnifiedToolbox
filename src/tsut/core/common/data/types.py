"""Define the base Data types for the TSUT Framework."""

from typing import Any

import numpy as np
import pandas as pd


class BaseData:
    """Base class for all data types in the TSUT Framework."""

    # INFO : For now there isn't any specifics that need to be defined across all data types.


class TimeSeries(BaseData):
    """Class representing time series data.

    The implementation is highly inspired from the Darts TimeSeries class. It is intentionally more lightweight however.
    All capabilities related to auto-pre-treatment have been removed as these are intended to be integrated as pipeline Nodes.
    """

    def __init__(
        self,
        times: pd.DatetimeIndex | pd.RangeIndex | pd.Index,
        values: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a 'TimeSeries' from a time index 'times' and a numpy array of 'values'."""
        # TODO : Implement TimeSeries class
