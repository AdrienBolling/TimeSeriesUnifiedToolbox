import importlib
import pkgutil

# Auto-discovery of all _register.py files in the components package and import them to register the nodes.
# This allows us to keep the registration of nodes organized in their respective modules, while still ensuring that all nodes are registered when the components package is imported.

def _auto_discovery() -> None:
    """Import component registration modules and invoke their ``register_nodes`` hooks."""
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

