"""Define the base Data types for the TSUT Framework."""

from typing import Any, NamedTuple, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

D = TypeVar("D", bound="Data")
K = TypeVar("K")


type ContextData = dict[str, Any] | BaseModel | NamedTuple


class Data:
    """Base class for all data types in the TSUT Framework."""

    # INFO : For now there isn't any specifics that need to be defined across all data types. It is intentional, this type is only used to ensure all inputs/outputs come from the TSUT library.


class TimeSeries(Data):
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
