"""Built-in component nodes for the TSUT framework.

This package provides ready-to-use implementations of data sources, models,
transforms, metrics, and sinks. All components are automatically discovered
and registered in the global :data:`~tsut.core.nodes.registry.node_registry.NODE_REGISTRY`
when the ``tsut`` package is imported.

To add a custom component, create a ``_register.py`` module in the appropriate
subdirectory with a ``register_nodes()`` function.
"""

import importlib
import pkgutil


def _auto_discovery() -> None:
    """Walk all subpackages and import every ``_register`` module.

    Each ``_register`` module is expected to define a ``register_nodes()``
    callable that registers its nodes with the global NODE_REGISTRY.
    """
    for module_info in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        if module_info.name.endswith("._register"):
            mod = importlib.import_module(module_info.name)
            if hasattr(mod, "register_nodes"):
                met = getattr(mod, "register_nodes")
                if callable(met):
                    met()
                    print(f"Successfully registered nodes from module: {module_info.name}")
            else:
                print(f"Warning: Module {module_info.name} does not have a register_nodes function. Skipping registration for this module.")


