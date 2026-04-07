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
    """Hyperparameter tuner backed by Ray Tune.

    Builds a trainable function from a pipeline configuration and delegates
    the search to Ray Tune, converting metrics to a scalar objective.
    """

    def __init__(self, pipeline: Pipeline, *, config: RayPipelineTunerConfig) -> None:
        """Initialize the RayPipelineTuner.

        Args:
            pipeline: A pipeline whose config will be used as the base for
                each trial.  The pipeline is compiled internally for
                validation.
            config: Tuner-level configuration.

        """
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
        """Build the default hyperparameter space from all tunable nodes.

        Keys follow the ``"node_name/hp_name"`` convention and values are
        Ray Tune search-space objects (e.g. ``tune.choice``,
        ``tune.uniform``).  The returned dict can be customised before
        passing it to :meth:`tune`.

        Returns:
            Dict of ``{node_name/hp_name: ray_tune_definition}``.

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
        """Run the hyperparameter tuning process via Ray Tune.

        Args:
            param_space: Search space dict (``node_name/hp_name`` keys).
            optimization_metric: Name of the single metric to optimise.
                Mutually exclusive with *metric_aggregator* (one must be
                provided).
            metric_aggregator: Callable that reduces the full metrics dict
                to a scalar.  Takes precedence over *optimization_metric*.
            tune_config: Ray Tune :class:`TuneConfig`.  Defaults are used
                when ``None``.
            run_config: Ray Tune :class:`RunConfig`.  Defaults are used
                when ``None``.

        Raises:
            ValueError: If neither *optimization_metric* nor
                *metric_aggregator* is provided.

        """
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
        """Return the best result from the most recent tuning run.

        Args:
            metric: Metric name to rank by.  Defaults to
                ``"optimization_metric"``.
            mode: ``"max"`` or ``"min"``.

        Returns:
            The best :class:`ray.tune.Result` object.

        Raises:
            ValueError: If :meth:`tune` has not been called yet.

        """
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
        """Build the trainable function passed to Ray Tune.

        Args:
            optimization_metric: Single metric name to extract.
            metric_aggregator: Callable reducing all metrics to a scalar.

        Returns:
            A function ``(config: dict) -> None`` compatible with
            :class:`ray.tune.Tuner`.

        """
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
        """Convert pipeline metrics to scalar floats for Ray Tune reporting.

        Args:
            metrics: Dict of metric node outputs as :class:`TabularData`.

        Returns:
            Dict mapping metric names to float values.

        """
        # For now, we will just convert any Data/DataContext objects to their "data" attribute if they have one, or to their string representation if they don't. This is a very naive implementation and should be improved in the future.
        converted_metrics = {}
        for metric_name, metric_data in metrics.items():
            converted_metrics[metric_name] = metric_data.to_numpy()[0][
                0, 0
            ]  # We assume the metrics are always TabularData with a single row and column, will need to be verifier in the future, maybe we can make a MetricData class.
        return converted_metrics

    def _convert_config(self, config: dict) -> PipelineConfig:
        """Convert a flat Ray Tune config dict into a :class:`PipelineConfig`.

        Args:
            config: Flat dict with ``"node_name/hp_name"`` keys and sampled
                values.

        Returns:
            A :class:`PipelineConfig` with the hyperparameters applied.

        """
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
