"""MLflow logging wrapper for any :class:`PipelineRunner`.

Wraps the ``train`` / ``evaluate`` / ``infer`` lifecycle with an MLflow run,
automatically logging:

* **Pipeline metadata** — name, version, node graph summary.
* **Pipeline config** — the full ``PipelineConfig`` JSON (includes
  hyperparameters and running configs for every node).
* **Pipeline parameters** — trained weights / learned state of every node,
  serialised as a JSON artifact after training.
* **Pipeline graph** — interactive HTML render of the DAG.
* **Metrics** — all scalar metrics returned by ``evaluate()``.
* **Timing** — wall-clock duration of ``train`` and ``evaluate`` phases.
"""

import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking.fluent import ActiveRun
from pydantic import Field
from pydantic_settings import BaseSettings

from tsut.core.common.data.data import ArrayLike, DataContext
from tsut.core.pipeline.runners.pipeline_runner import PipelineRunner


class MLFlowLoggerWrapperConfig(BaseSettings):
    """Configuration for the :class:`MLFlowLoggerWrapper`.

    Fields prefixed with ``MLFLOW_`` can be loaded from environment
    variables (e.g. ``MLFLOW_TRACKING_URI``, ``MLFLOW_EXPERIMENT_NAME``).
    Explicit constructor values take precedence over env vars.
    """

    model_config = {"env_prefix": "MLFLOW_"}

    tracking_uri: str | None = Field(
        default=None,
        description=(
            "MLflow tracking server URI. "
            "``None`` uses the default local ``./mlruns`` directory. "
            "Env: ``MLFLOW_TRACKING_URI``."
        ),
    )
    experiment_name: str = Field(
        default="tsut",
        description="MLflow experiment name. Created automatically if it does not exist. Env: ``MLFLOW_EXPERIMENT_NAME``.",
    )
    run_name: str | None = Field(
        default=None,
        description="Optional human-readable name for the MLflow run. Env: ``MLFLOW_RUN_NAME``.",
    )
    nested: bool = Field(
        default=False,
        description=(
            "If ``True``, the run is created as a nested child of the "
            "currently active run (useful inside Ray Tune trials). "
            "Env: ``MLFLOW_NESTED``."
        ),
    )
    log_pipeline_params: bool = Field(
        default=True,
        description="Log the pipeline's trained parameters (learned weights) as a JSON artifact after training. Env: ``MLFLOW_LOG_PIPELINE_PARAMS``.",
    )
    log_pipeline_config: bool = Field(
        default=True,
        description="Log the full ``PipelineConfig`` JSON as an MLflow artifact. Env: ``MLFLOW_LOG_PIPELINE_CONFIG``.",
    )
    log_pipeline_graph: bool = Field(
        default=True,
        description="Log an interactive HTML render of the pipeline DAG as an MLflow artifact. Env: ``MLFLOW_LOG_PIPELINE_GRAPH``.",
    )
    log_train_time: bool = Field(
        default=False,
        description="Log ``train_duration_s`` as an MLflow metric after training. Env: ``MLFLOW_LOG_TRAIN_TIME``.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Extra user-defined tags added to every MLflow run.",
    )


class MLFlowLoggerWrapper:
    """Wrap a :class:`PipelineRunner` with MLflow lifecycle logging.

    Usage::

        runner = SmartRunner(pipeline, config=SmartRunnerConfig())
        mlflow_runner = MLFlowLoggerWrapper(
            runner,
            config=MLFlowLoggerWrapperConfig(experiment_name="my_experiment"),
        )
        mlflow_runner.train()
        metrics = mlflow_runner.evaluate()
        # Everything is logged in MLflow automatically.
    """

    def __init__(
        self,
        pipeline_runner: PipelineRunner,
        *,
        config: MLFlowLoggerWrapperConfig | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            pipeline_runner: The runner instance to wrap.
            config: MLflow logging configuration.  Uses defaults when
                ``None``.

        """
        self._runner = pipeline_runner
        self._config = config or MLFlowLoggerWrapperConfig()
        self._run_id: str | None = None

        if self._config.tracking_uri is not None:
            mlflow.set_tracking_uri(self._config.tracking_uri)
        mlflow.set_experiment(self._config.experiment_name)

    # --- Public properties ------------------------------------------------

    @property
    def runner(self) -> PipelineRunner:
        """The underlying pipeline runner."""
        return self._runner

    @property
    def run_id(self) -> str | None:
        """MLflow run ID of the most recent run (``None`` before ``train``)."""
        return self._run_id

    # --- Lifecycle methods ------------------------------------------------

    def train(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]] | None = None,
    ) -> None:
        """Start an MLflow run, log params, train, and log timing.

        Args:
            input_data: Optional external data forwarded to the runner.

        """
        with mlflow.start_run(
            run_name=self._config.run_name,
            nested=self._config.nested,
        ) as run:
            self._run_id = run.info.run_id
            self._log_pipeline_metadata()

            if self._config.log_pipeline_config:
                self._log_pipeline_config_artifact()

            if self._config.log_pipeline_graph:
                self._log_pipeline_graph_artifact()

            if self._config.log_train_time:
                t0 = time.perf_counter()
                self._runner.train(input_data=input_data)
                duration_s = time.perf_counter() - t0
                mlflow.log_metric("train_duration_s", duration_s)
            else:
                self._runner.train(input_data=input_data)

            if self._config.log_pipeline_params:
                self._log_pipeline_params_artifact()

    def evaluate(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]] | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Evaluate the pipeline and log all resulting metrics.

        If a run was started by :meth:`train`, metrics are appended to
        that same run. Otherwise a new run is created.

        Args:
            input_data: Optional external data forwarded to the runner.

        Returns:
            The raw metrics mapping returned by the runner.

        """
        ctx = self._resume_or_start_run()
        with ctx as run:
            self._run_id = run.info.run_id

            t0 = time.perf_counter()
            metrics = self._runner.evaluate(input_data=input_data)
            duration_s = time.perf_counter() - t0

            mlflow.log_metric("evaluate_duration_s", duration_s)
            self._log_metrics(dict(metrics))

        return metrics

    def infer(
        self,
        input_data: Mapping[str, Mapping[str, tuple[ArrayLike, DataContext]]] | None = None,
    ) -> Mapping[str, tuple[ArrayLike, DataContext]]:
        """Run inference via the underlying runner (no MLflow logging).

        Args:
            input_data: Optional external data forwarded to the runner.

        Returns:
            Sink-node outputs from the runner.

        """
        return self._runner.infer(input_data=input_data)

    # --- Delegation shortcuts ---------------------------------------------

    def get_params(self) -> dict[str, dict[str, Any]]:
        """Delegate to the underlying runner."""
        return self._runner.get_params()

    def set_params(self, params: dict[str, dict[str, Any]]) -> None:
        """Delegate to the underlying runner."""
        self._runner.set_params(params=params)

    def save_params_to_dir(self, dir_path: str) -> None:
        """Delegate to the underlying runner."""
        self._runner.save_params_to_dir(dir_path=dir_path)

    def load_params_from_dir(self, dir_path: str) -> None:
        """Delegate to the underlying runner."""
        self._runner.load_params_from_dir(dir_path=dir_path)

    # --- Internal helpers -------------------------------------------------

    def _resume_or_start_run(self) -> ActiveRun:
        """Return a context manager that resumes the existing run or starts a new one."""
        if self._run_id is not None:
            return mlflow.start_run(
                run_id=self._run_id,
                nested=self._config.nested,
            )
        return mlflow.start_run(
            run_name=self._config.run_name,
            nested=self._config.nested,
        )

    def _log_pipeline_metadata(self) -> None:
        """Log pipeline name, version, node list, and user-defined tags."""
        pipeline = self._runner.pipeline
        mlflow.set_tag("pipeline_name", pipeline.name)
        mlflow.set_tag("pipeline_version", str(pipeline.version))
        node_names = list(pipeline.node_objects.keys())
        mlflow.set_tag("pipeline_nodes", ", ".join(node_names))

        for key, value in self._config.tags.items():
            mlflow.set_tag(key, value)

    def _log_pipeline_params_artifact(self) -> None:
        """Save the pipeline's trained parameters (learned weights) as a pickle artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._runner.save_params_to_dir(tmpdir)
            for path in Path(tmpdir).iterdir():
                mlflow.log_artifact(str(path))

    def _log_pipeline_graph_artifact(self) -> None:
        """Render the pipeline DAG as an interactive HTML artifact."""
        pipeline = self._runner.pipeline
        html = pipeline.render_to_html(full_html=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline_graph.html"
            path.write_text(html)
            mlflow.log_artifact(str(path))

    def _log_pipeline_config_artifact(self) -> None:
        """Dump the full PipelineConfig as a JSON artifact."""
        pipeline = self._runner.pipeline
        config_json = pipeline.config.model_dump_json(indent=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline_config.json"
            path.write_text(config_json)
            mlflow.log_artifact(str(path))

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        """Convert runner metrics to scalars and log them to MLflow.

        Args:
            metrics: Dict mapping metric names to ``(array, context)`` tuples
                returned by the runner's ``evaluate``. Only single-cell
                arrays (a single scalar) are logged as MLflow metrics;
                other shapes are skipped silently.

        """
        flat: dict[str, float] = {}
        for name, value in metrics.items():
            scalar = self._extract_scalar(value)
            if scalar is not None:
                flat[name] = scalar
        if flat:
            mlflow.log_metrics(flat)

    @staticmethod
    def _extract_scalar(value: Any) -> float | None:
        """Return a single float from a ``(array, context)`` metric tuple.

        Handles pandas, numpy, and torch arrays via ``np.asarray`` and
        returns ``None`` for anything that is not a 1-element array.
        """
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, tuple) and len(value) == 2:  # (array, context)  # noqa: PLR2004
            array, _ = value
            if isinstance(array, pd.DataFrame):
                array = array.to_numpy()
            arr = np.asarray(array)
            if arr.size == 1:
                return float(arr.reshape(()))
        return None
