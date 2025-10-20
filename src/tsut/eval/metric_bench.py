from metrics import registry as metrics_registry


class MetricBench:
    """
    A class implementing a benchmark of metrics.
    """

    def __init__(self, metric_names, model_metadata) -> None:
        self.metric_names = metric_names
        self.model_metadata = model_metadata
