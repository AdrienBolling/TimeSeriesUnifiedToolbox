"""FeatureConcatenate operation node, which concatenates the features of two input dataframes along the column axis (features)."""

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


class FeatureConcatenateMetadata(TransformMetadata):
    """Metadata for the FeatureConcatenate TransformNode in a TSUT Pipeline."""

    node_name: str = "FeatureConcatenate"
    input_type: str = "pd.DataFrame"
    output_type: str = "pd.DataFrame"
    description: str = "Concatenate multiple DataFrames along the column axis (features)."
    trainable: bool = False  # This transform is not trainable, since it does not learn any parameters from the data. It simply applies a deterministic operation to the input data.

class FeatureConcatenateRunningConfig(TransformRunningConfig):
    """Running configuration for the FeatureConcatenate TransformNode in the TSUT Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the transform itself, rather for how to run it.
    For example, in some transforms, this could be very specific parameters such as the backend to use for computations etc.
    """

    matching_number_samples: bool = True  # Whether to check that the number of samples (rows) in the input DataFrames are the same before concatenating. If True, an error will be raised if the number of samples do not match. If False, the concatenation will be performed without checking, which may result in NaN values for the missing samples in the shorter DataFrame.

class FeatureConcatenateHyperParameters(TransformHyperParameters):
    """Hyperparameters for the FeatureConcatenate TransformNode in the TSUT Framework.

    This will usually be used for parameters that are relevant for the definition of the transform itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in some transforms, this could be the window size for a rolling window transform, etc.
    """

class FeatureConcatenateConfig(TransformConfig[FeatureConcatenateRunningConfig, FeatureConcatenateHyperParameters]):
    """Configuration for the FeatureConcatenate TransformNode in the TSUT Framework."""

    running_config: FeatureConcatenateRunningConfig = FeatureConcatenateRunningConfig()
    hyperparameters: FeatureConcatenateHyperParameters = FeatureConcatenateHyperParameters()
    in_ports: dict[str, Port] = {"input_1": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch feature1", desc="First input DataFrame to concatenate"), "input_2": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch feature2", desc="Second input DataFrame to concatenate")}
    out_ports: dict[str, Port] = {"output": Port(arr_type=ArrayLikeEnum.PANDAS, data_category=DataCategoryEnum.MIXED, data_shape="batch feature1+feature2" if running_config.matching_number_samples else "_ feature1+feature2", desc="Concatenated output DataFrame")}

class FeatureConcatenate(TransformNode[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext, None]):
    """Node that concatenates multiple DataFrames along the column axis (features)."""

    metadata = FeatureConcatenateMetadata()

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialize the FeatureConcatenateNode with the given configuration."""
        self._config = config

    def fit(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> None:
        """Don't fit.

        Since it does not learn any parameters from the data. It simply applies a deterministic operation to the input data.
        """

    def transform(self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Concatenate the input DataFrames along the column axis (features)."""
        df1, ctx1 = data["input_1"]
        df2, ctx2 = data["input_2"]

        if self._config.running_config.matching_number_samples and df1.shape[0] != df2.shape[0]:
            raise ValueError(f"Number of samples (rows) in the input DataFrames do not match: {df1.shape[0]} vs {df2.shape[0]}. Set matching_number_samples to False in the running config to concatenate without checking.")

        concatenated_df = pd.concat([df1, df2], axis=1)

        # Concatenate the contexts.
        output_ctx = self._concatenate_contexts(ctx1, ctx2)

        return {"output": (concatenated_df, output_ctx)}

    # --- Internal helper methods ---

    def _concatenate_contexts(self, ctx1: TabularDataContext, ctx2: TabularDataContext) -> TabularDataContext:
        """Concatenate the contexts of the input DataFrames.

        This is a simple implementation that assumes that the contexts are compatible and can be concatenated by simply merging their metadata. In a more complex implementation, you may want to add checks for compatibility and handle conflicts in the metadata (e.g., if both contexts have a "feature_types" metadata, you may want to check that they are compatible and merge them accordingly).
        """
        return TabularDataContext(
            columns= ctx1.columns + ctx2.columns,
            dtypes= ctx1.dtypes + ctx2.dtypes,
            categories= ctx1.categories + ctx2.categories,
        )
