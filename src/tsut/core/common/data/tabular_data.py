"""Define the base Data types for the TSUT Framework."""

from enum import StrEnum

import jaxtyping
import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
import torch
from dataclasses import dataclass
from IPython.display import display, HTML
from tsut.core.common.data.data import Data, DataContext, ArrayLike, ArrayLikeEnum, DataCategoryEnum, CategoricalData, NumericalData, MixedData, DATA_CATEGORY_MAPPING, DataCategory, INVERSE_DATA_CATEGORY_MAPPING

# Constants for internal data shape validation
_TABULAR_NDIM = 2  # (batch, feature)

@dataclass
class TabularDataContext(DataContext):
    """(column names, dtypes, categories), this is the context of the data which is passed along the nodes."""

    columns: list[str]  # We use a default factory to avoid mutable default arguments issues with dataclasses 
    dtypes: list[np.dtype]  # We use a default factory to avoid mutable default arguments issues with dataclasses
    categories: list[type[DataCategory]]

    @property
    def dump_dict(self) -> dict[str, list[str]]:
        """Dump the TabularDataContext to a dictionary for serialization."""
        return {
            "columns": self.columns,
            "dtypes": [str(dtype) for dtype in self.dtypes],  # We can only serialize the name of the dtype, since it's a type.
            "categories": [str(INVERSE_DATA_CATEGORY_MAPPING[cat]) for cat in self.categories],  # We can only serialize the name of the category, since it's a type.
        }

    @property
    def dump_tuple(self) -> tuple[list[str], list[np.dtype], list[type[DataCategory]]]:
        """Dump the TabularDataContext to a tuple for serialization."""
        return (self.columns, self.dtypes, self.categories)

    def remove_columns(self, columns_to_remove: list[str]) -> None:
        """Remove columns from the context."""
        # Find the indexes of the columns to remove
        indexes_to_remove = [self.columns.index(col) for col in columns_to_remove if col in self.columns]
        # Remove the columns, dtypes, and categories at the corresponding indexes
        self.columns = [col for i, col in enumerate(self.columns) if i not in indexes_to_remove]
        self.dtypes = [dtype for i, dtype in enumerate(self.dtypes) if i not in indexes_to_remove]
        self.categories = [cat for i, cat in enumerate(self.categories) if i not in indexes_to_remove]


def tabular_context_from_dict_dump(dump_dict: dict[str, list[str]]) -> TabularDataContext:
    """Load the TabularDataContext from a dictionary."""
    return TabularDataContext(
        columns=dump_dict["columns"],
        dtypes=[np.dtype(dtype_str) for dtype_str in dump_dict["dtypes"]],
        categories=[DATA_CATEGORY_MAPPING[DataCategoryEnum(cat_str)] for cat_str in dump_dict["categories"]]
    )

 # INFO : For now there isn't any specifics that need to be defined across all data types. It is intentional, this type is only used to ensure all inputs/outputs come from the TSUT library.

class TabularData(Data):
    """Common data type for Tabular Data in the TSUT framework. This type is only supposed to be used by the Pipeline, it is not expected that the user interacts with it directly.

    Basically a wrapper around some pandas data, can be easily cast from and to several other data types.
    Schemas are enforced for the dimensions order using jaxtyping and beartypes.

    In particular, for tabular data, the order is (B, F) -> (Batch, Feature)

    For example to access the j-th feature of the i-th sample of a batch, we can do data[i, j].

    Any data is considered a batch, at worst a batch of 1 or 0 samples. This will make it easier to reshape and convert in the long run.

    Since in the background the data is a pandas dataframe, there is also the matter of colum names and dtypes. In cases where the target conversion type does not natively handle these, a tuple will be returned as follows :
    (data_array, column_names, dtypes). It is up to the user to preserve the order of columns, and pass them along as needed (which should be quite easy, since the Shuffling transform node should be the only one modifying order).
    Finally, we explictely attach the category of each feature, this is gonna be useful because pandas tends to implicitely convert data types as it sees fit, and some model are not supposed to ingest categorical or numerical data.
    """

    def __init__(
        self,
        data: ArrayLike,
        columns: list[str],
        dtypes: list[np.dtype],
        categories: list[type[DataCategory]] | None = None,
        *,
        infer_categories: bool = True,  # Whether to infer categories from dtypes if categories are not provided. This is a bit hacky but it should work for most cases. We can always allow the user to explicitly pass the categories if they want to be more precise.
    ) -> None:
        """Initialize the TabularData."""
        # Config
        self._infer_categories = infer_categories
        if data is not None:
            self._validate_data(data, columns, dtypes, categories)
        if isinstance(data, pd.DataFrame):
            self.from_pandas(data, categories)
        elif categories is None:
            raise ValueError("Categories must be provided for numpy and torch data, or infer_categories must be set to True to infer categories from dtypes.")
        elif isinstance(data, np.ndarray):
            self.from_numpy(data, columns, dtypes, categories)
        elif isinstance(data, torch.Tensor):
            self.from_tensor(data, columns, dtypes, categories)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    # --- Convenience API ---
    @property
    def data(self) -> tuple[ArrayLike | None, list[str] | None, list[np.dtype] | None, list[type[DataCategory]] | None]:
        """Return the underlying data."""
        return self._data, self._columns, self._dtypes, self._categories

    @property
    def is_initialized(self) -> bool:
        """Return whether the data is initialized."""
        return self._data is not None and self._columns is not None and self._dtypes is not None and self._categories is not None

    # --- Pipeline compatibility check methods ---

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data."""
        return self._data.shape

    @property
    def dtype(self) -> str:
        """Return "mixed_data" if the data has mixed categories, otherwise return the category of the data."""
        if self._categories is None:
            raise ValueError("Data is not initialized yet.")
        if all(cat == self._categories[0] for cat in self._categories):
            return str(INVERSE_DATA_CATEGORY_MAPPING[self._categories[0]])
        return str(DataCategoryEnum.MIXED)

    @property
    def dtypes(self) -> list[np.dtype]:
        """Return the dtypes of the data."""
        return self._dtypes

    @property
    def columns(self) -> list[str]:
        """Return the column names of the data."""
        return self._columns

    @property
    def categories(self) -> list[type[DataCategory]]:
        """Return the categories of the data."""
        if self._categories is None:
            raise ValueError("Data is not initialized yet.")
        return self._categories

    @property
    def category(self) -> type[DataCategory]:
        """Return the category of the data.

        This is an aggregate, if all features are of the same category, return that category, otherwise return mixed.
        """
        if self._categories is None:
            raise ValueError("Data is not initialized yet.")
        if all(cat == self._categories[0] for cat in self._categories):
            return self._categories[0]
        return MixedData

    # --- Validation methods ---

    def _validate_data(self, data: ArrayLike | None, columns: list[str] | None, dtypes: list[np.dtype] | None, categories: list[type[DataCategory]] | None) -> bool:
        """Validate the schema of a pandas DataFrame for TabularData."""
        # Check that the data is 2D, has column names, and that the dtypes can be inferred.
        if data is not None and data.ndim != _TABULAR_NDIM:
            raise ValueError(f"Data must be {_TABULAR_NDIM}D, but got {data.ndim}D.")
        if columns is None:
            raise ValueError("Columns must be provided.")
        if dtypes is None:
            raise ValueError("Dtypes must be provided.")
        if categories is None:
            raise ValueError("Categories must be provided.")
        if data is not None:
            if len(columns) != data.shape[1]:
                raise ValueError(f"Number of columns must match data shape, but got {len(columns)} columns and data with shape {data.shape}.")
            if len(dtypes) != data.shape[1]:
                raise ValueError(f"Number of dtypes must match data shape, but got {len(dtypes)} dtypes and data with shape {data.shape}.")
            if len(categories) != data.shape[1]:
                raise ValueError(f"Number of categories must match data shape, but got {len(categories)} categories and data with shape {data.shape}.")
        return True

    # --- Conversion FROM methods ---
    def from_pandas(self, data: pd.DataFrame, categories: list[type[DataCategory]] | None = None) -> None:
        """Initialize the TabularData from a pandas DataFrame."""
        columns = data.columns.tolist()
        dtypes = data.dtypes.tolist()

        if categories is None and self._infer_categories:
            # Infer categories from dtypes, this is a bit hacky but it should work for most cases. We can always allow the user to explicitly pass the categories if they want to be more precise.
            inferred_categories = []
            for dtype in dtypes:
                if pd.api.types.is_numeric_dtype(dtype):
                    inferred_categories.append(NumericalData)
                elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype):
                    inferred_categories.append(CategoricalData)
                else:
                    inferred_categories.append(MixedData)  # If we can't infer, we consider it mixed, which is the most permissive category.
            categories = inferred_categories
        if categories is not None and len(categories) != len(columns):
            raise ValueError("Length of categories must match number of columns.")

        self._validate_data(data, columns, dtypes, categories)

        self._data = data
        self._columns = columns
        self._dtypes = dtypes
        self._categories = categories

    def from_numpy(self, data: np.ndarray, columns: list[str], dtypes: list[np.dtype], categories: list[type[DataCategory]]) -> None:
        """Initialize the TabularData from a numpy array."""
        self._validate_data(data, columns, dtypes, categories)
        self._data = pd.DataFrame(data, columns=columns,).astype(dict(zip(columns, dtypes, strict=True)))  # Convert to pandas DataFrame and set column names and dtypes
        self._columns = columns
        self._dtypes = dtypes
        self._categories = categories

    def from_tensor(self, data: torch.Tensor, columns: list[str], dtypes: list[np.dtype], categories: list[type[DataCategory]]) -> None:
        """Initialize the TabularData from a torch tensor."""
        self.from_numpy(data.cpu().numpy(), columns, dtypes, categories)

    # --- Conversion TO methods ---
    def to_pandas(self) -> tuple[pd.DataFrame, TabularDataContext]:
        """Convert the TabularData to a pandas DataFrame."""
        if not self.is_initialized:
            raise ValueError("Data is not initialized yet.")
        if self._categories is None: # Typechecker stuff, could be removed without issues (tbi, do we really lose performance)
            raise ValueError("Data is not initialized yet.")
        return (self._data, TabularDataContext(self._columns, self._dtypes, self._categories))

    def to_numpy(self) -> tuple[np.ndarray, TabularDataContext]:
        """Convert the TabularData to a numpy array, along with column names, dtypes, and categories."""
        if not self.is_initialized:
            raise ValueError("Data is not initialized yet.")
        if self._categories is None: # Typechecker stuff, could be removed without issues
            raise ValueError("Data is not initialized yet.")
        return (self._data.values, TabularDataContext(self._columns, self._dtypes, self._categories))

    def to_tensor(self) -> tuple[torch.Tensor, TabularDataContext]:
        """Convert the TabularData to a torch tensor, along with column names, dtypes, and categories."""
        if not self.is_initialized:
            raise ValueError("Data is not initialized yet.")
        if self._categories is None: # Typechecker stuff, could be removed without issues
            raise ValueError("Data is not initialized yet.")
        return (torch.from_numpy(self._data.values), TabularDataContext(self._columns, self._dtypes, self._categories))



    # --- Overloading basic functions for convenience ---
    def __str__(self) -> str:
        """Represent the TabularData as a pandas array with multi-index columns for column names, dtypes, and categories."""
        col_names = [str(col) for col in self._columns]
        dtype_names = [str(dt) for dt in self._dtypes]
        category_names = [str(INVERSE_DATA_CATEGORY_MAPPING[cat]) for cat in self._categories] if self._categories is not None else ["unknown" for _ in col_names]

        # Create a DataFrame only for display purposes
        df = pd.DataFrame(
            self._data.to_numpy(),
            columns=pd.MultiIndex.from_arrays(
                [col_names, dtype_names, category_names],
                names=["column", "dtype", "category"],
            ),
        ) # Display the DataFrame as HTML for better formatting in Jupyter notebooks
        return f"{self.__class__.__name__}\n{df.__str__()}(\nShape: {self.shape})"

    def _repr_html_(self):
        """Represent the TabularData as a pandas array with multi-index columns for column names, dtypes, and categories, in HTML format for better display in Jupyter notebooks."""
        col_names = [str(col) for col in self._columns]
        dtype_names = [str(dt) for dt in self._dtypes]
        category_names = [str(INVERSE_DATA_CATEGORY_MAPPING[cat]) for cat in self._categories] if self._categories is not None else ["unknown" for _ in col_names]

        # Create a DataFrame only for display purposes
        df = pd.DataFrame(
            self._data.to_numpy(),
            columns=pd.MultiIndex.from_arrays(
                [col_names, dtype_names, category_names],
                names=["column", "dtype", "category"],
            ),
        )
        html = f"<h3>{self.__class__.__name__} (Shape: {self.shape})</h3>"
        html += df._repr_html_()
        return html