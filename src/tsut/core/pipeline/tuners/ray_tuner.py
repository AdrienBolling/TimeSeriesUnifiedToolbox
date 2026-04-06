"""Ray Pipeline Tuner module."""

import tempfile
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from ray import tune
from ray.tune import RunConfig, TuneConfig

from tsut.core.common.data.data import TabularData
from tsut.core.common.typechecking.typeguards import (
    has_hyperparameter_space,
    has_hyperparameters_config,
)
from tsut.core.pipeline.pipeline import Pipeline, PipelineConfig
from tsut.core.pipeline.runners.smart_runner import (
    SmartRunnerConfig,
    TabularSmartRunner,
)


class RayPipelineTunerConfig(BaseModel):
    """Configuration for the RayPipelineTuner."""

    runner_config: SmartRunnerConfig = SmartRunnerConfig()


class RayPipelineTuner:
    """Ray Pipeline Tuner implementation for the TSUT Framework."""

    def __init__(self, pipeline: Pipeline, *, config: RayPipelineTunerConfig) -> None:
        """Initialize the RayPipelineTuner with a pipeline and configuration."""
        self._pipeline_config = pipeline.config  # We do this because this way the config will have been fully instantiated in the Pipeline
        self._config = config
        self._results_grid = None

        # Dummy Pipeline, for validation and node objects instantiation.
        self._dummy_pipe = Pipeline(config=self._pipeline_config)
        self._dummy_pipe.compile()

    # --- Public API ---

    @property
    def tuner(self) -> tune.Tuner | None:
        """Get the Ray Tune Tuner object after tuning has been run."""
        return self._tuner

    def default_hyperparameter_space(self) -> dict[str, Any]:
        """Get the default hyperparameter space for tuning.

        This method should return a dictionary where the keys are in the format "node_name/hp_name" and the values are Ray Tune compatible hyperparameter definitions. (e.g. tune.choice([1, 2, 3]), tune.uniform(0, 1), etc.)
        The user can then modify this default hyperparameter space according to their needs before passing it to the "tune" method.
        """
        hp_space = {}
        for node_name, node_obj in self._dummy_pipe.node_objects.items():
            if has_hyperparameter_space(node_obj):
                for hp_name, hp_value in node_obj.hyperparameter_space.items():
                    hp_space[f"{node_name}/{hp_name}"] = hp_value
        return hp_space

    def tune(
        self,
        param_space: dict[str, Any],
        optimization_metric: str | None = None,
        metric_aggregator: Callable[[dict[str, float]], float] | None = None,
        tune_config: TuneConfig | None = None,
        run_config: RunConfig | None = None,
    ) -> None:
        """Run the tuning process on the underlying pipeline runner."""
        if tune_config is None:
            tune_config = TuneConfig()
        if run_config is None:
            run_config = RunConfig()
        trainable = self._trainable(
            optimization_metric=optimization_metric, metric_aggregator=metric_aggregator
        )

        # Force the "optimization_metric" key to the one we want. This is used by Ray Tune to determine which metric to optimize for.
        tune_config.metric = "optimization_metric"

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        self._tuner = tuner
        tuner.fit()

    def get_best(self, metric: str | None = None, mode: str = "max") -> dict[str, Any]:
        """Get the best hyperparameters found during tuning."""
        if self._tuner is None:
            msg = "Tuning has not been run yet. Please run the tune() method first."
            raise ValueError(msg)
        best_result = self._tuner.get_best_result(metric=metric, mode=mode)
        return best_result

    # --- Internal methods ---

    def _trainable(
        self,
        optimization_metric: str | None = None,
        metric_aggregator: Callable[[dict[str, float]], float] | None = None,
    ) -> Callable[[dict], None]:
        """Get the function to be used as the trainable for Ray Tune."""
        fn = None
        if metric_aggregator is not None:
            fn = metric_aggregator
        elif optimization_metric is not None:
            fn = lambda metrics: metrics[optimization_metric]  # noqa: E731
        else:
            msg = "Either optimization_metric or metric_aggregator must be provided. In the event both are provided, the metric_aggregator will be used."
            raise ValueError(msg)

        def trainable(config: dict) -> None:
            # Set the parameters of the underlying pipeline runner according to the config provided by Ray Tune.
            # The config passed by Ray Tune would be a dictionnary where the keys are hyperparameters, we need to convert it to a PipelineConfig ourselves.
            pipe_conf = self._convert_config(config)
            pipe = Pipeline(config=pipe_conf)
            pipe.compile()
            runner = TabularSmartRunner(
                pipeline=pipe, config=self._config.runner_config
            )

            # Load the parameters of the pipeline if they exist. This would allow for resuming tuning from a checkpoint.
            # this is quite useless until we start using real neural networks. But it is good to have it for the future.
            if tune.get_checkpoint():
                checkpoint = tune.get_checkpoint()
                with checkpoint.as_directory() as checkpoint_dir:
                    # Load the state of the runner from the checkpoint directory if it exists. This would allow for resuming training from a checkpoint.
                    runner.load_params_from_dir(checkpoint_dir)
            # Train the pipeline.
            runner.train()
            # Evaluate the pipeline and get the metrics.
            metrics = runner.evaluate()
            # Convert the metrics to a format that can be returned to Ray Tune. (e.g. convert any Data/DataContext objects to dictionaries or other serializable formats)
            metrics = self._convert_metrics(metrics)
            metrics["optimization_metric"] = fn(metrics)
            # Report the metrics to Ray Tune. This is only done once as we are not doing any actual training loop here
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save the state of the runner to the temporary directory. This would allow for resuming training from a checkpoint in the future.
                runner.save_params_to_dir(tmpdir)
                # Create a checkpoint from the temporary directory and report it to Ray Tune along with the metrics.
                checkpoint = tune.Checkpoint.from_directory(tmpdir)
                tune.report(metrics=metrics, checkpoint=checkpoint)

        return trainable

    def _convert_metrics(self, metrics: dict[str, TabularData]) -> dict[str, float]:
        """Convert the metrics returned by the pipeline runner to a format that can be reported to Ray Tune."""
        # For now, we will just convert any Data/DataContext objects to their "data" attribute if they have one, or to their string representation if they don't. This is a very naive implementation and should be improved in the future.
        converted_metrics = {}
        for metric_name, metric_data in metrics.items():
            converted_metrics[metric_name] = metric_data.to_numpy()[0][
                0, 0
            ]  # We assume the metrics are always TabularData with a single row and column, will need to be verifier in the future, maybe we can make a MetricData class.
        return converted_metrics

    def _convert_config(self, config: dict) -> PipelineConfig:
        """Convert the config provided by Ray Tune to a PipelineConfig."""
        # We know that the config dict provided by Ray Tune will be a flat dict where the keys are the hyperparameters. We need to convert it to a PipelineConfig ourselves.
        # To build this hyperparameter_space, the user will have used self.default_hyperparameter_space() and then modified it according to their needs.
        # this default hp_dict is formated as follows :
        # {"node_name/hp_name": tune_compatible_definition}  # noqa: ERA001
        # So what we will receive in the "config" argument will be :
        # {"node_name/hp_name": value}  # noqa: ERA001

        config_dict = {}
        for key, value in config.items():
            node_name, hp_name = key.split("/")
            if node_name not in config_dict:
                config_dict[node_name] = {}
            config_dict[node_name][hp_name] = value

        pipe_conf = self._pipeline_config.model_copy()
        dummy_pipe = Pipeline(
            config=pipe_conf
        )  # We create a dummy pipeline to benefit from the config validation logic.
        for node_name, hp_dict in config_dict.items():
            for hp_name, hp_value in hp_dict.items():
                conf = dummy_pipe.internal_config.nodes[node_name][1]
                if has_hyperparameters_config(conf):
                    conf.model_copy(
                        update={
                            "hyperparameters": conf.hyperparameters.model_copy(
                                update={hp_name: hp_value}
                            )
                        }
                    )
        return pipe_conf
