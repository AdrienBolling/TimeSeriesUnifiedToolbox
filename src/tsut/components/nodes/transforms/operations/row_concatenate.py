"""Module for the RowConcatenate operation node."""

import pandas as pd

from tsut.core.common.data.data import ArrayLikeEnum, DataCategoryEnum
from tsut.core.common.data.tabular_data import TabularDataContext
from tsut.core.nodes.node import Port
from tsut.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)


class RowConcatenateMetadata(TransformMetadata):
    """Metadata for the RowConcatenate TransformNode in a TSUT Pipeline."""

    node_name: str = "RowConcatenate"
    input_type: str = "pd.DataFrame"
    output_type: str = "pd.DataFrame"
    description: str = "Concatenate multiple DataFrames along the row axis (index)."
    trainable: bool = False  # This transform is not trainable, since it does not learn any parameters from the data. It simply applies a deterministic operation to the input data.

class RowConcatenateRunningConfig(TransformRunningConfig):
    """Running configuration for the RowConcatenate TransformNode in the TSUT Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the transform itself, rather for how to run it.
    For example, in some transforms, this could be very specific parameters such as the backend to use for computations etc.
    """

class RowConcatenateHyperParameters(TransformHyperParameters):
    """Hyperparameters for the RowConcatenate TransformNode in the TSUT Framework.

    This will usually be used for parameters that are relevant for the definition of the transform itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in some transforms, this could be the window size for a rolling window transform, etc.
    """

class RowConcatenateConfig(TransformConfig[RowConcatenateRunningConfig, RowConcatenateHyperParameters]):
    """Configuration for the RowConcatenate TransformNode in the TSUT Framework."""

    running_config: RowConcatenateRunningConfig = RowConcatenateRunningConfig()
    hyperparameters: RowConcatenateHyperParameters = RowConcatenateHyperParameters()
    in_ports: dict[str, Port] = {"input_1": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch1 feature",desc="First input DataFrame to concatenate"), "input_2": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch2 feature", desc="Second input DataFrame to concatenate")}
    out_ports: dict[str, Port] = {"output": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch1+batch2 feature", desc="Concatenated output DataFrame")}

class RowConcatenate(TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, None]):
    """Node that concatenates multiple DataFrames along the row axis (index)."""

    metadata = RowConcatenateMetadata()


    def __init__(self, *, config: TransformConfig) -> None:
        """Initialize the row concatenation transform with its runtime configuration."""
        self._config = config

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Don't fit.

        Since it does not learn any parameters from the data. It simply applies a deterministic operation to the input data.
        """

    def transform(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Concatenate the input DataFrames along the row axis (index)."""
        df1, ctx1 = data["input_1"]
        df2, ctx2 = data["input_2"]

        if not self._check_context_compatibility(ctx1, ctx2):
            raise ValueError("The contexts of the input DataFrames are not compatible for concatenation. Please ensure that the contexts are compatible before concatenating.")

        # Concatenate the DataFrames along the row axis (index)
        concatenated_df = pd.concat([df1, df2], axis=0)

        # For the context, we can simply take the context of the first input, or we could implement a more complex logic to merge the contexts if needed. For simplicity, we'll take the context of the first input.
        return {"output": (concatenated_df, ctx1)}

    # --- Internal methods

    def _check_context_compatibility(self, ctx1: TabularDataContext, ctx2: TabularDataContext) -> bool:
        """Check the compatibility of the contexts of the two input DataFrames.

        Here this means checking if the two contexts are stricly equal, which is a very strict condition but it is the safest one to ensure that we are not concatenating data with incompatible contexts. In the future, we could implement a more flexible logic to check for compatibility of contexts, for example by checking if they have the same features, even if they have different metadata for these features.
        """
        return ctx1 == ctx2
