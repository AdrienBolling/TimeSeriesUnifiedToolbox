from torchmetrics import MeanSquaredError
from tsut.core.nodes.metrics.metric_node import MetricNode, MetricNodeConfig, MetricNodeRunningConfig, MetricNodeMetadata
import torch
from tsut.core.common.data.tabular_data import TabularDataContext, tabular_context_from_dict_dump
from tsut.core.common.data.data import ArrayLikeEnum, DataCategoryEnum
from tsut.core.nodes.node import Port
import numpy as np

class MSEMetadata(MetricNodeMetadata):
    """Metadata for the MSE Metric Node."""
    node_name: str = "MSE"
    node_description: str = "Mean Squared Error Metric Node. This node computes the Mean Squared Error between the true and predicted values."

class MSERunningConfig(MetricNodeRunningConfig):
    """Running configuration for the MSE Metric Node."""
    pass

class MSEConfig(MetricNodeConfig):
    """Configuration for the MSE Metric Node."""
    running_config: MSERunningConfig = MSERunningConfig()
    in_ports: dict[str, Port] = {
        "preds": Port(arr_type=ArrayLikeEnum.TORCH, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch features", mode=["evaluation"], desc="Predicted values."),
        "target": Port(arr_type=ArrayLikeEnum.TORCH, data_category=DataCategoryEnum.NUMERICAL, data_shape="batch features", mode=["evaluation"], desc="True values.")
    }
    out_ports: dict[str, Port] = {
        "mse": Port(arr_type=ArrayLikeEnum.TORCH, data_category=DataCategoryEnum.NUMERICAL, data_shape="scalar", mode=["evaluation"], desc="Computed Mean Squared Error.")
    }

class MSENode(MetricNode[torch.Tensor, TabularDataContext, torch.Tensor, TabularDataContext]):
    """MSE Metric Node."""

    metadata = MSEMetadata()
    _trainable: bool = False

    def __init__(self, *, config: MSEConfig) -> None:
        """Initialize the MSE Metric Node with the given configuration."""
        self._config = config
        self._metric = MeanSquaredError()

    def update(self, data: dict[str, tuple[torch.Tensor, TabularDataContext]]) -> None:
        """Update the MSE metric with the given data."""
        preds, _ = data["preds"]
        target, target_ctx = data["target"]
        self._metric = MeanSquaredError(num_outputs=target.shape[1])  # Re-instantiate the metric to reset its state before each update, as we want to compute the MSE for each batch independently
        self._metric.update(preds=preds, target=target)

        self._ctx = target_ctx  # Store the context for the output

    def compute(self) -> dict[str, tuple[torch.Tensor, TabularDataContext]]:
        """Compute the MSE metric with the given data."""
        mse_value = self._metric.compute()
        cols = [f"{col_name}_mse" for col_name in self._ctx.columns]
        dtypes = ["float32" for _ in self._ctx.columns]
        categories = [str(DataCategoryEnum.NUMERICAL) for _ in self._ctx.columns]
        ctx = tabular_context_from_dict_dump({
            "columns": cols,
            "dtypes": dtypes,
            "categories": categories
        })
        return {"mse": (mse_value.unsqueeze(0), ctx)}
