"""Node factory utilities for the TSUT framework.

Provides helper functions to wrap common ML objects (sklearn estimators,
torch ``nn.Module`` subclasses, plain callables) into fully registered TSUT
pipeline nodes with minimal boilerplate.

Example — wrapping an sklearn estimator::

    from tsut.components.utils.factory import create_sklearn_model_node, PortSpec

    create_sklearn_model_node(
        "Ridge",
        Ridge,
        in_ports={
            "X": PortSpec("numpy", "numerical", "batch features"),
            "y": PortSpec("numpy", "numerical", "batch targets",
                          mode=["training", "evaluation"]),
        },
        out_ports={"pred": PortSpec("numpy", "numerical", "batch targets")},
        hyperparameters={"alpha": (float, 1.0)},
    )

Example — wrapping a torch nn.Module::

    from tsut.components.utils.factory import (
        create_torch_model_node, PortSpec, TorchTrainingSpec,
    )

    create_torch_model_node(
        "MyMLP",
        MyMLPModule,
        in_ports={
            "X": PortSpec("numpy", "numerical", "batch features"),
            "y": PortSpec("numpy", "numerical", "batch targets",
                          mode=["training", "evaluation"]),
        },
        out_ports={"pred": PortSpec("numpy", "numerical", "batch targets")},
        training=TorchTrainingSpec(
            optimizer_class=torch.optim.Adam,
            loss_fn=nn.MSELoss(),
            epochs=50,
            batch_size=32,
        ),
        hyperparameters={"input_dim": (int, 10), "hidden_dim": (int, 64)},
    )
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from pydantic import Field as PydanticField
from pydantic import create_model
from torch import nn

from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
    TabularDataContext,
    tabular_context_from_dict_dump,
)
from tsut.core.common.enums import NodeExecutionMode
from tsut.core.common.typechecking.typeguards import (
    is_valid_torch_module,
)
from tsut.core.nodes.models.model import (
    Model,
    ModelConfig,
    ModelHyperParameters,
    ModelMetadata,
    ModelRunningConfig,
)
from tsut.core.nodes.node import Port
from tsut.core.nodes.registry.node_registry import NODE_REGISTRY
from tsut.core.nodes.transform.transform import (
    TransformConfig,
    TransformHyperParameters,
    TransformMetadata,
    TransformNode,
    TransformRunningConfig,
)

# ============================================================
# User-facing data classes
# ============================================================

_ARR_MAP: dict[str, ArrayLikeEnum] = {
    "numpy": ArrayLikeEnum.NUMPY,
    "pandas": ArrayLikeEnum.PANDAS,
    "torch": ArrayLikeEnum.TORCH,
}

_CAT_MAP: dict[str, DataCategoryEnum] = {
    "numerical": DataCategoryEnum.NUMERICAL,
    "categorical": DataCategoryEnum.CATEGORICAL,
    "mixed": DataCategoryEnum.MIXED,
}

_STRUCT_MAP: dict[str, DataStructureEnum] = {
    "Data": DataStructureEnum.DATA,
    "TabularData": DataStructureEnum.TABULAR,
}

_MODE_MAP: dict[str, str] = {
    "training": NodeExecutionMode.TRAINING,
    "inference": NodeExecutionMode.INFERENCE,
    "evaluation": NodeExecutionMode.EVALUATION,
    "all": NodeExecutionMode.ALL,
}


@dataclass
class PortSpec:
    """Simplified port specification for factory-created nodes.

    Args:
        arr_type: Array backend — ``"numpy"``, ``"pandas"``, or ``"torch"``.
        data_category: ``"numerical"``, ``"categorical"``, or ``"mixed"``.
        data_shape: Jaxtyping shape string, e.g. ``"batch features"``.
        desc: Human-readable description of the port.
        data_structure: ``"Data"`` or ``"TabularData"`` (default).
        optional: Whether the port may be left unconnected.
        mode: Execution modes in which this port is active.
            Defaults to ``["all"]``.

    """

    arr_type: str
    data_category: str
    data_shape: str
    desc: str = ""
    data_structure: str = "TabularData"
    optional: bool = False
    mode: list[str] | None = None

    def to_port(self) -> Port:
        """Convert to a full :class:`~tsut.core.nodes.node.Port`."""
        return Port(
            arr_type=_ARR_MAP[self.arr_type],
            data_category=_CAT_MAP[self.data_category],
            data_structure=_STRUCT_MAP.get(
                self.data_structure, DataStructureEnum.TABULAR
            ),
            data_shape=self.data_shape,
            desc=self.desc,
            optional=self.optional,
            mode=[_MODE_MAP.get(m, m) for m in (self.mode or ["all"])],
        )


@dataclass
class TorchTrainingSpec:
    """Training loop configuration for torch ``nn.Module`` wrapping.

    Args:
        optimizer_class: A ``torch.optim`` class (e.g. ``torch.optim.Adam``).
        optimizer_kwargs: Keyword arguments forwarded to the optimizer
            constructor.  Defaults to ``{"lr": 1e-3}``.
        loss_fn: A loss module instance (e.g. ``nn.MSELoss()``) or any
            callable ``(pred, target) -> scalar``.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.  ``None`` means full-batch.
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"`` (picks the
            best available accelerator).

    """

    optimizer_class: type = torch.optim.Adam
    optimizer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-3},
    )
    loss_fn: Any = field(default_factory=nn.MSELoss)
    epochs: int = 10
    batch_size: int | None = None
    device: str = "cpu"


# ============================================================
# Internal helpers
# ============================================================


def _build_port_dict(specs: dict[str, PortSpec]) -> dict[str, Port]:
    """Convert a mapping of :class:`PortSpec` to TSUT :class:`Port` objects."""
    return {name: spec.to_port() for name, spec in specs.items()}


def _build_pydantic_model(
    class_name: str,
    base_class: type,
    fields: dict[str, tuple[type, Any]] | None,
) -> type:
    """Dynamically create a Pydantic model subclass.

    Args:
        class_name: Name for the generated class.
        base_class: Pydantic ``BaseModel`` subclass to inherit from.
        fields: Mapping of ``{field_name: (type, default)}``.  ``None``
            produces an empty subclass.

    Returns:
        A new Pydantic model class.

    """
    if not fields:
        return create_model(class_name, __base__=base_class)
    field_defs: dict[str, Any] = {}
    for fname, (ftype, fdefault) in fields.items():
        field_defs[fname] = (ftype, PydanticField(default=fdefault))
    return create_model(class_name, __base__=base_class, **field_defs)


def _introspect_sklearn_params(
    estimator_class: type,
) -> dict[str, tuple[type, Any]]:
    """Extract ``(param_name -> (type, default))`` from an sklearn estimator.

    Args:
        estimator_class: An sklearn estimator class.

    Returns:
        A dict suitable for passing as ``hyperparameters`` to the factory.

    """
    sig = inspect.signature(estimator_class.__init__)
    params: dict[str, tuple[type, Any]] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        ptype = param.annotation if param.annotation != inspect.Parameter.empty else Any
        default = param.default if param.default != inspect.Parameter.empty else None
        params[pname] = (ptype, default)
    return params


def _resolve_device(device: str) -> torch.device:
    """Resolve a device string to a ``torch.device``.

    ``"auto"`` picks CUDA if available, then MPS, then CPU.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ============================================================
# Sklearn model factory
# ============================================================


def create_sklearn_model_node(
    name: str,
    estimator_class: type,
    *,
    in_ports: dict[str, PortSpec],
    out_ports: dict[str, PortSpec],
    hyperparameters: dict[str, tuple[type, Any]] | None = None,
    running_config: dict[str, tuple[type, Any]] | None = None,
    description: str = "",
    auto_register: bool = True,
) -> type[Model]:
    """Create a TSUT Model node wrapping an sklearn estimator.

    The generated node delegates ``fit`` → ``estimator.fit`` and
    ``predict`` → ``estimator.predict``.  Constructor kwargs are split
    between *hyperparameters* (tunable) and *running_config* (fixed).
    If neither is supplied, all sklearn ``__init__`` params are treated
    as hyperparameters.

    Args:
        name: Registry name for the node (e.g. ``"Ridge"``).
        estimator_class: An sklearn estimator class (not an instance).
        in_ports: Input port specifications.  By convention the feature
            port is named ``"X"`` and the target port ``"y"``.
        out_ports: Output port specifications.
        hyperparameters: ``{param: (type, default)}`` for tunable params.
            ``None`` auto-introspects from *estimator_class*.
        running_config: ``{param: (type, default)}`` for fixed params.
        description: Human-readable description for the node metadata.
        auto_register: Whether to register in ``NODE_REGISTRY``.

    Returns:
        The dynamically created ``Model`` subclass.

    """
    # -- Resolve hyperparameters ----------------------------------------
    if hyperparameters is None and running_config is None:
        hyperparameters = _introspect_sklearn_params(estimator_class)

    hp_cls = _build_pydantic_model(
        f"{name}HyperParameters",
        ModelHyperParameters,
        hyperparameters,
    )
    rc_cls = _build_pydantic_model(
        f"{name}RunningConfig",
        ModelRunningConfig,
        running_config,
    )

    # -- Metadata -------------------------------------------------------
    metadata_cls = type(
        f"{name}Metadata",
        (ModelMetadata,),
        {
            "__annotations__": {"node_name": str, "description": str},
            "node_name": name,
            "description": (
                description or f"Auto-generated wrapper for {estimator_class.__name__}."
            ),
        },
    )

    # -- Config ---------------------------------------------------------
    in_port_dict = _build_port_dict(in_ports)
    out_port_dict = _build_port_dict(out_ports)

    config_cls = create_model(
        f"{name}Config",
        __base__=ModelConfig[hp_cls, rc_cls],
        hyperparameters=(hp_cls, PydanticField(default_factory=hp_cls)),
        running_config=(rc_cls, PydanticField(default_factory=rc_cls)),
        in_ports=(
            dict[str, Port],
            PydanticField(default=in_port_dict),
        ),
        out_ports=(
            dict[str, Port],
            PydanticField(default=out_port_dict),
        ),
    )

    # -- Node class -----------------------------------------------------
    in_port_names = list(in_ports)
    out_port_names = list(out_ports)

    # Capture variables in a closure-safe way
    _estimator_class = estimator_class
    _config_cls = config_cls
    _metadata_instance = metadata_cls()
    _in_port_names = in_port_names
    _out_port_names = out_port_names

    class _SklearnNode(
        Model[
            np.ndarray,
            TabularDataContext,
            np.ndarray,
            TabularDataContext,
            dict[str, Any],
        ]
    ):
        metadata = _metadata_instance

        def __init__(self, *, config: _config_cls) -> None:  # type: ignore[valid-type]
            self._config = config
            all_params = {
                **config.hyperparameters.model_dump(),
                **config.running_config.model_dump(),
            }
            self._model = _estimator_class(**all_params)
            self._target_context_dump: dict[str, list[str]] = {}

        def fit(
            self,
            data: dict[str, tuple[np.ndarray, TabularDataContext]],
        ) -> None:
            """Fit the underlying sklearn estimator."""
            x_arr, _ = data[_in_port_names[0]]
            if "y" in data:
                y_arr, y_ctx = data["y"]
                self._model.fit(x_arr, y_arr)
                self._target_context_dump = y_ctx.dump_dict
            else:
                self._model.fit(x_arr)

        def predict(
            self,
            data: dict[str, tuple[np.ndarray, TabularDataContext]],
        ) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
            """Predict using the underlying sklearn estimator."""
            x_arr, _ = data[_in_port_names[0]]
            pred: np.ndarray = self._model.predict(x_arr)
            if pred.ndim == 1:
                pred = pred[:, np.newaxis]
            pred_ctx = tabular_context_from_dict_dump(
                self._target_context_dump,
            )
            return {_out_port_names[0]: (pred, pred_ctx)}

        def get_params(self) -> dict[str, Any]:
            """Return serialisable state."""
            return {
                "model_params": self._model.get_params(),
                "target_context": self._target_context_dump,
            }

        def set_params(self, params: dict[str, Any]) -> None:
            """Restore from a previously serialised state."""
            self._model.set_params(**params["model_params"])
            self._target_context_dump = params["target_context"]

    _SklearnNode.__name__ = f"{name}Node"
    _SklearnNode.__qualname__ = f"{name}Node"

    # -- Register -------------------------------------------------------
    if auto_register:
        NODE_REGISTRY.register(
            name=name,
            node_class=_SklearnNode,
            node_config_class=config_cls,
            running_config_class=rc_cls,
            hyperparameters_class=hp_cls,
        )

    return _SklearnNode


# ============================================================
# Torch model factory
# ============================================================


def create_torch_model_node(
    name: str,
    module_class: type[nn.Module],
    *,
    in_ports: dict[str, PortSpec],
    out_ports: dict[str, PortSpec],
    training: TorchTrainingSpec,
    hyperparameters: dict[str, tuple[type, Any]] | None = None,
    running_config: dict[str, tuple[type, Any]] | None = None,
    description: str = "",
    auto_register: bool = True,
) -> type[Model]:
    """Create a TSUT Model node wrapping a torch ``nn.Module``.

    The module class **must** define a custom ``train`` method that
    contains the training logic.  During ``fit``, the generated node
    delegates to ``module.train(data)`` where *data* is the port
    dictionary received by the node.  A ``TypeError`` is raised at
    factory time if no custom ``train`` method is found on the class.

    During ``predict``, the node performs a standard forward pass.

    Args:
        name: Registry name for the node.
        module_class: A ``torch.nn.Module`` subclass (not an instance).
            Must define a custom ``train(self, data)`` method.
            Its ``__init__`` must accept the keys defined in
            *hyperparameters* as keyword arguments.
        in_ports: Input port specifications.
        out_ports: Output port specifications.
        training: Training loop configuration.
        hyperparameters: ``{param: (type, default)}`` forwarded to
            ``module_class.__init__``.
        running_config: ``{param: (type, default)}`` for non-tunable
            training knobs.  ``epochs``, ``batch_size``, and ``device``
            are injected automatically from *training*.
        description: Human-readable description.
        auto_register: Whether to register in ``NODE_REGISTRY``.

    Returns:
        The dynamically created ``Model`` subclass.

    """
    # -- Validate that module_class defines a custom train method ----------
    if "train_module" not in vars(module_class):
        msg = (
            f"{module_class.__name__} does not define a custom `train_module` method. "
            f"The torch factory requires the module class to implement its own "
            f"`train(self, data)` method containing the training logic."
        )
        raise TypeError(msg)

    # -- Resolve running config with training spec defaults ---------------
    rc_fields: dict[str, tuple[type, Any]] = {
        "epochs": (int, training.epochs),
        "batch_size": (int | None, training.batch_size),
        "device": (str, training.device),
    }
    if running_config:
        rc_fields.update(running_config)

    hp_cls = _build_pydantic_model(
        f"{name}HyperParameters",
        ModelHyperParameters,
        hyperparameters,
    )
    rc_cls = _build_pydantic_model(
        f"{name}RunningConfig",
        ModelRunningConfig,
        rc_fields,
    )

    metadata_cls = type(
        f"{name}Metadata",
        (ModelMetadata,),
        {
            "__annotations__": {"node_name": str, "description": str},
            "node_name": name,
            "description": (
                description
                or f"Auto-generated torch wrapper for {module_class.__name__}."
            ),
        },
    )

    in_port_dict = _build_port_dict(in_ports)
    out_port_dict = _build_port_dict(out_ports)

    config_cls = create_model(
        f"{name}Config",
        __base__=ModelConfig[hp_cls, rc_cls],
        hyperparameters=(hp_cls, PydanticField(default_factory=hp_cls)),
        running_config=(rc_cls, PydanticField(default_factory=rc_cls)),
        in_ports=(
            dict[str, Port],
            PydanticField(default=in_port_dict),
        ),
        out_ports=(
            dict[str, Port],
            PydanticField(default=out_port_dict),
        ),
    )

    in_port_names = list(in_ports)
    out_port_names = list(out_ports)

    # Capture in closure
    _module_class = module_class
    _config_cls = config_cls
    _metadata_instance = metadata_cls()
    _optimizer_class = training.optimizer_class
    _optimizer_kwargs = training.optimizer_kwargs
    _loss_fn = training.loss_fn
    _in_port_names = in_port_names
    _out_port_names = out_port_names

    class _TorchNode(
        Model[
            np.ndarray,
            TabularDataContext,
            np.ndarray,
            TabularDataContext,
            dict[str, Any],
        ]
    ):
        metadata = _metadata_instance

        def __init__(self, *, config: _config_cls) -> None:  # type: ignore[valid-type]
            self._config = config
            hp_dict = config.hyperparameters.model_dump()
            self._device = _resolve_device(config.running_config.device)
            self._module = _module_class(**hp_dict).to(self._device)
            self._optimizer = _optimizer_class(
                self._module.parameters(),
                **_optimizer_kwargs,
            )
            self._loss_fn = _loss_fn
            self._epochs: int = config.running_config.epochs
            self._batch_size: int | None = config.running_config.batch_size
            self._target_context_dump: dict[str, list[str]] = {}

        def fit(
            self,
            data: dict[str, tuple[np.ndarray, TabularDataContext]],
        ) -> None:
            """Delegate training to the module's custom ``train`` method."""
            _, y_ctx = data.get("y", (None, None))
            if y_ctx is not None:
                self._target_context_dump = y_ctx.dump_dict
            if is_valid_torch_module(self._module):
                self._module.train_module(data)

        def predict(
            self,
            data: dict[str, tuple[np.ndarray, TabularDataContext]],
        ) -> dict[str, tuple[np.ndarray, TabularDataContext]]:
            """Run a forward pass and return predictions."""
            x_raw, _ = data[_in_port_names[0]]
            x_t = torch.as_tensor(x_raw, dtype=torch.float32).to(self._device)

            self._module.eval()
            with torch.no_grad():
                pred = self._module(x_t)

            pred_np = pred.cpu().numpy()
            if pred_np.ndim == 1:
                pred_np = pred_np[:, np.newaxis]
            pred_ctx = tabular_context_from_dict_dump(
                self._target_context_dump,
            )
            return {_out_port_names[0]: (pred_np, pred_ctx)}

        def get_params(self) -> dict[str, Any]:
            """Return serialisable model state."""
            return {
                "model_state_dict": {
                    k: v.cpu() for k, v in self._module.state_dict().items()
                },
                "optimizer_state_dict": self._optimizer.state_dict(),
                "target_context": self._target_context_dump,
            }

        def set_params(self, params: dict[str, Any]) -> None:
            """Restore from a previously serialised state."""
            self._module.load_state_dict(params["model_state_dict"])
            self._module.to(self._device)
            self._optimizer.load_state_dict(params["optimizer_state_dict"])
            self._target_context_dump = params["target_context"]

    _TorchNode.__name__ = f"{name}Node"
    _TorchNode.__qualname__ = f"{name}Node"

    if auto_register:
        NODE_REGISTRY.register(
            name=name,
            node_class=_TorchNode,
            node_config_class=config_cls,
            running_config_class=rc_cls,
            hyperparameters_class=hp_cls,
        )

    return _TorchNode


# ============================================================
# Callable transform factory
# ============================================================


def create_callable_transform_node(
    name: str,
    transform_fn: Callable[
        [dict[str, tuple[Any, TabularDataContext]], dict[str, Any]],
        dict[str, tuple[Any, TabularDataContext]],
    ],
    *,
    fit_fn: Callable[[dict[str, tuple[Any, TabularDataContext]]], dict[str, Any]]
    | None = None,
    in_ports: dict[str, PortSpec],
    out_ports: dict[str, PortSpec],
    hyperparameters: dict[str, tuple[type, Any]] | None = None,
    running_config: dict[str, tuple[type, Any]] | None = None,
    description: str = "",
    auto_register: bool = True,
) -> type[TransformNode]:
    """Create a TSUT TransformNode wrapping plain callables.

    Args:
        name: Registry name for the node.
        transform_fn: ``(data, state) -> output_data``.  *data* is the
            standard port dict, *state* is whatever ``fit_fn`` returned.
        fit_fn: ``(data) -> state``.  ``None`` for stateless transforms
            (the state dict will be empty).
        in_ports: Input port specifications.
        out_ports: Output port specifications.
        hyperparameters: ``{param: (type, default)}`` for tunable params.
        running_config: ``{param: (type, default)}`` for fixed params.
        description: Human-readable description.
        auto_register: Whether to register in ``NODE_REGISTRY``.

    Returns:
        The dynamically created ``TransformNode`` subclass.

    """
    hp_cls = _build_pydantic_model(
        f"{name}HyperParameters",
        TransformHyperParameters,
        hyperparameters,
    )
    rc_cls = _build_pydantic_model(
        f"{name}RunningConfig",
        TransformRunningConfig,
        running_config,
    )

    metadata_cls = type(
        f"{name}Metadata",
        (TransformMetadata,),
        {
            "__annotations__": {"node_name": str, "description": str},
            "node_name": name,
            "description": description or f"Auto-generated transform node '{name}'.",
        },
    )

    in_port_dict = _build_port_dict(in_ports)
    out_port_dict = _build_port_dict(out_ports)

    config_cls = create_model(
        f"{name}Config",
        __base__=TransformConfig[hp_cls, rc_cls],
        hyperparameters=(hp_cls, PydanticField(default_factory=hp_cls)),
        running_config=(rc_cls, PydanticField(default_factory=rc_cls)),
        in_ports=(
            dict[str, Port],
            PydanticField(default=in_port_dict),
        ),
        out_ports=(
            dict[str, Port],
            PydanticField(default=out_port_dict),
        ),
    )

    _config_cls = config_cls
    _metadata_instance = metadata_cls()
    _fit_fn = fit_fn
    _transform_fn = transform_fn

    class _CallableTransform(
        TransformNode[Any, TabularDataContext, Any, TabularDataContext, dict[str, Any]]
    ):
        metadata = _metadata_instance

        def __init__(self, *, config: _config_cls) -> None:  # type: ignore[valid-type]
            self._config = config
            self._fitted = False
            self._state: dict[str, Any] = {}

        def fit(
            self,
            data: dict[str, tuple[Any, TabularDataContext]],
        ) -> None:
            """Delegate to the user-provided fit callable."""
            if _fit_fn is not None:
                self._state = _fit_fn(data) or {}

        def transform(
            self,
            data: dict[str, tuple[Any, TabularDataContext]],
        ) -> dict[str, tuple[Any, TabularDataContext]]:
            """Delegate to the user-provided transform callable."""
            return _transform_fn(data, self._state)

        def get_params(self) -> dict[str, Any]:
            """Return the fitted state dict."""
            return self._state

        def set_params(self, params: dict[str, Any]) -> None:
            """Restore from a previously serialised state."""
            self._state = params
            self._fitted = True

    _CallableTransform.__name__ = f"{name}Node"
    _CallableTransform.__qualname__ = f"{name}Node"

    if auto_register:
        NODE_REGISTRY.register(
            name=name,
            node_class=_CallableTransform,
            node_config_class=config_cls,
            running_config_class=rc_cls,
            hyperparameters_class=hp_cls,
        )

    return _CallableTransform
