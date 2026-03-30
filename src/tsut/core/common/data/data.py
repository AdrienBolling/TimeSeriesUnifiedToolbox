"""Module for the base types of data in the TSUT Framework."""

from enum import StrEnum
from typing import TypeVar

import jaxtyping
import numpy as np
import pandas as pd
import torch
from pydantic.dataclasses import dataclass


class Data:
    """Base class for all data types in the TSUT Framework."""

@dataclass
class DataContext:
    """Base class for all data contexts in the TSUT Framework."""

type ArrayLike = pd.DataFrame | np.ndarray | torch.Tensor
class ArrayLikeEnum(StrEnum):
    """Enum for array-like data types. This is used for the arr_type field in the Port model to specify the type of data array that a port accepts or outputs."""

    PANDAS = "pd.DataFrame"
    NUMPY = "np.ndarray"
    TORCH = "torch.Tensor"
ARRAY_MAPPING = {
    ArrayLikeEnum.PANDAS: pd.DataFrame,
    ArrayLikeEnum.NUMPY: np.ndarray,
    ArrayLikeEnum.TORCH: torch.Tensor,
}
# --- Jaxtyping things
class CategoricalData(jaxtyping.AbstractDtype):
    """Generic dtype for jaxtyping. This is used to bypass the dtype check in jaxtyping, since we want to allow any dtype for our data types, and we will handle dtype validation ourselves in the data classes.

    We will be using jaxtyping only for shape checking.
    """

    dtypes = ["categorical_data"]

    def __str__(self) -> str:
        """Represent the CategoricalData dtype as a string."""
        return "categorical_data"

class NumericalData(jaxtyping.AbstractDtype):
    """Generic dtype for jaxtyping. This is used to bypass the dtype check in jaxtyping, since we want to allow any dtype for our data types, and we will handle dtype validation ourselves in the data classes.

    We will be using jaxtyping only for shape checking.
    """

    dtypes = ["numerical_data"]

    def __str__(self) -> str:
        """Represent the NumericalData dtype as a string."""
        return "numerical_data"

class MixedData(jaxtyping.AbstractDtype):
    """Used to check if the category of data are compatible between nodes through jaxtyping."""

    dtypes = ["categorical_data", "numerical_data", "mixed_data"]

    def __str__(self) -> str:
        """Represent the MixedData dtype as a string."""
        return "mixed_data"

DataCategory = CategoricalData | NumericalData | MixedData
class DataCategoryEnum(StrEnum):
    """Enum for data categories. (used for pydantic model dumping and loading from json)."""

    CATEGORICAL = "categorical_data"
    NUMERICAL = "numerical_data"
    MIXED = "mixed_data"
DATA_CATEGORY_MAPPING = {
    DataCategoryEnum.CATEGORICAL: CategoricalData,
    DataCategoryEnum.NUMERICAL: NumericalData,
    DataCategoryEnum.MIXED: MixedData,
}
INVERSE_DATA_CATEGORY_MAPPING = {v: k for k, v in DATA_CATEGORY_MAPPING.items()}


# --- TypeVars for the data types and contexts, to be used in the Nodes and subnodes definition
D_I = TypeVar("D_I", bound=ArrayLike)  # Input data type for the Nodes
D_O = TypeVar("D_O", bound=ArrayLike)  # Output data type for the Nodes
D_C_I = TypeVar("D_C_I", bound=DataContext)  # Data context type for the Nodes (used for the input data context)
D_C_O = TypeVar("D_C_O", bound=DataContext)  # Data context type for the Nodes (used for the output data context)
P = TypeVar("P", bound=dict)  # To be used for the parameters's type in the Nodes, if needed. This is optional and can be set to None if not used.
