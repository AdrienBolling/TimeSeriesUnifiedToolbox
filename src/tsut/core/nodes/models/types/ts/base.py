"""Define base class for Time Series models in the TSUT Framework."""

from abc import ABC
from typing import TypeVar

from tsut.core.common.data.types import Data, TimeSeries
from tsut.core.nodes.models.base import Model, ModelConfig


class TimeSeriesModelConfig(ModelConfig):
    """Base metadata configuration for all Time Series Models in the TSUT Framework."""

    output_shape: tuple[int, ...] | None = None
    intput_data: type = TimeSeries


TS_D = TypeVar("TS_D", bound=TimeSeries)
D_O = TypeVar("D_O", bound=Data)


class TimeSeriesModel[TS_D, D_O](
    Model[TS_D, D_O], ABC
):  # TimeSeriesModel is already implicitly abstract but explicit is better.
    """Define the base class for Time Series models in the TSUT Framework."""

