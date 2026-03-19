import inspect
from typing import Any

# ============================================================
# Helpers
# ============================================================

def _in_ipython_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return "IPKernelApp" in shell.config
    except Exception:
        return False


def _is_class_like(obj: Any) -> bool:
    return inspect.isclass(obj)


def _safe_name(obj: Any) -> str:
    return getattr(obj, "__name__", type(obj).__name__)


def _safe_qualname(obj: Any) -> str:
    return getattr(obj, "__qualname__", _safe_name(obj))


def _safe_module(obj: Any) -> str:
    return getattr(obj, "__module__", "")


def _safe_doc_first_line(obj: Any) -> str:
    try:
        doc = inspect.getdoc(obj)
        if not doc:
            return ""
        return doc.splitlines()[0].strip()
    except Exception:
        return ""


def _safe_file(obj: Any) -> str:
    try:
        return inspect.getsourcefile(obj) or ""
    except Exception:
        return ""


def _safe_lineno(obj: Any) -> int | None:
    try:
        _, lineno = inspect.getsourcelines(obj)
        return lineno
    except Exception:
        return None


def _serialize_value(value: Any) -> Any:
    """Convert values into something display-friendly.

    Rules:
    - classes get expanded into readable metadata
    - lists/tuples/sets become strings
    - dicts become repr strings
    - scalars pass through
    """
    if _is_class_like(value):
        return {
            "__kind__": "class",
            "display": _safe_name(value),
            "qualname": _safe_qualname(value),
            "module": _safe_module(value),
            "doc": _safe_doc_first_line(value),
            "file": _safe_file(value),
            "line": _safe_lineno(value),
            "object": value,
        }

    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))

    if isinstance(value, dict):
        return repr(value)

    return value


def _format_plain_value(value: Any) -> str:
    if isinstance(value, dict) and value.get("__kind__") == "class":
        return str(value.get("display", ""))
    return str(value)


def _clip(s: str, width: int) -> str:
    return s if len(s) <= width else s[: max(0, width - 1)] + "…"


# ============================================================
# Registry
# ============================================================

class NodeRegistry():
    """Registry for node definitions.

    The registry stores entries like:

        {
            "my_node": {
                "node_class": MyNode,
                "config_class": MyConfig,
                "family": "vision",
                "tags": ["train", "gpu"],
                "version": "v1",
            }
        }

    It is deliberately schema-flexible:
    any extra field you add to a node will automatically appear in the overview.
    """

    def __init__(self):
        self._registry: dict[str, dict[str, Any]] = {}

    # --------------------------------------------------------
    # Core CRUD
    # --------------------------------------------------------

    def register(
        self,
        name: str,
        node_class: type,
        node_config_class: type,
        **extra_fields: Any,
    ) -> None:
        """Public register method.

        Example:
            registry.register(
                "resize",
                ResizeNode,
                ResizeConfig,
                family="vision",
                tags=["image", "preprocess"],
                stable=True,
            )

        """
        if name in self._registry:
            message = (
                f"Node '{name}' is already registered. Please choose a different "
                f"name or unregister the existing node first."
            )
            raise ValueError(message)

        self._registry[name] = {
            "node_class": node_class,
            "config_class": node_config_class,
            **extra_fields,
        }

    def register_entry(self, name: str, **fields: Any) -> None:
        """Fully generic registration.

        Example:
            registry.register_entry(
                "resize",
                node_class=ResizeNode,
                config_class=ResizeConfig,
                family="vision",
                tags=["image", "preprocess"],
            )

        """
        if name in self._registry:
            message = (
                f"Node '{name}' is already registered. Please choose a different "
                f"name or unregister the existing node first."
            )
            raise ValueError(message)

        self._registry[name] = dict(fields)

    def unregister(self, name: str) -> None:
        if name not in self._registry:
            message = (
                f"Node '{name}' is not registered. Cannot unregister a non-existent node."
            )
            raise ValueError(message)
        del self._registry[name]

    def get(self, name: str) -> dict[str, Any]:
        if name not in self._registry:
            message = (
                f"Node '{name}' is not registered. Please register the node before trying to retrieve it."
            )
            raise ValueError(message)
        return self._registry[name]

    def __getitem__(self, name: str) -> dict[str, Any]:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def __str__(self) -> str:
        return f"NodeRegistry with {len(self._registry)} registered nodes: {list(self._registry.keys())}"

    def __repr__(self) -> str:
        return f"NodeRegistry(registry={self._registry})"

    def keys(self) -> list[str]:
        return list(self._registry.keys())

    def items(self):
        return self._registry.items()

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    def _rows(self) -> list[dict[str, Any]]:
        """Flatten registry entries into display rows.

        Adapts automatically to arbitrary fields.
        """
        rows: list[dict[str, Any]] = []

        for name, fields in self._registry.items():
            row: dict[str, Any] = {"name": name}

            for field_name, value in fields.items():
                serialized = _serialize_value(value)

                # Always keep a readable top-level column
                row[field_name] = (
                    serialized["display"]
                    if isinstance(serialized, dict) and serialized.get("__kind__") == "class"
                    else serialized
                )

                # If the field is a class, also expose helpful metadata columns
                if isinstance(serialized, dict) and serialized.get("__kind__") == "class":
                    row[f"{field_name}__module"] = serialized["module"]
                    row[f"{field_name}__qualname"] = serialized["qualname"]
                    row[f"{field_name}__doc"] = serialized["doc"]
                    row[f"{field_name}__file"] = serialized["file"]
                    row[f"{field_name}__line"] = serialized["line"]
                    row[f"{field_name}__object"] = serialized["object"]

            rows.append(row)

        return rows

    def _display_columns(self) -> list[str]:
        """Select user-facing columns dynamically.

        Rules:
        - show 'name' first
        - show original fields next
        - hide internal object columns
        - keep metadata columns available for filtering/sorting if needed
        """
        rows = self._rows()
        if not rows:
            return ["name"]

        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        preferred = ["name"]

        # original fields first (non-metadata)
        direct_fields = sorted(
            c for c in all_columns
            if c != "name" and "__" not in c
        )
        preferred.extend(direct_fields)

        # then useful metadata
        metadata_fields = sorted(
            c for c in all_columns
            if "__" in c and not c.endswith("__object")
        )
        preferred.extend(metadata_fields)

        return preferred

    # --------------------------------------------------------
    # Plain text rendering
    # --------------------------------------------------------

    def _plain_text_table(
        self,
        columns: list[str] | None = None,
        max_width: int = 140,
        max_col_width: int = 32,
    ) -> str:
        rows = self._rows()
        if not rows:
            return "<empty registry>"

        if columns is None:
            columns = self._display_columns()

        display_rows: list[dict[str, str]] = []
        for row in rows:
            display_rows.append(
                {col: _format_plain_value(row.get(col, "")) for col in columns}
            )

        widths = {}
        for col in columns:
            widths[col] = min(
                max(len(col), max(len(r[col]) for r in display_rows)),
                max_col_width,
            )

        total_width = sum(widths.values()) + 3 * (len(columns) - 1)
        if total_width > max_width:
            # crude shrink pass, enough for terminal fallback
            for col in sorted(columns, key=lambda c: widths[c], reverse=True):
                if total_width <= max_width:
                    break
                shrink_by = min(widths[col] - 8, total_width - max_width)
                if shrink_by > 0:
                    widths[col] -= shrink_by
                    total_width -= shrink_by

        header = " | ".join(col.ljust(widths[col]) for col in columns)
        sep = "-+-".join("-" * widths[col] for col in columns)

        lines = [header, sep]
        for row in display_rows:
            lines.append(
                " | ".join(_clip(row[col], widths[col]).ljust(widths[col]) for col in columns)
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # Notebook rendering
    # --------------------------------------------------------

    def _list_notebook(
        self,
        columns: list[str] | None = None,
        include_metadata: bool = False,
        include_source_inspector: bool = True,
    ):
        import ipywidgets as widgets
        import pandas as pd
        from IPython.display import HTML, display

        rows = self._rows()
        if not rows:
            display(HTML("<b>Registry is empty.</b>"))
            return None

        df = pd.DataFrame(rows)

        if columns is None:
            columns = self._display_columns()

        if not include_metadata:
            columns = [c for c in columns if "__" not in c]

        # only keep visible columns that exist
        columns = [c for c in columns if c in df.columns]
        if "name" not in columns:
            columns = ["name"] + columns

        # searchable columns: anything except raw object refs
        searchable_columns = [c for c in df.columns if not c.endswith("__object")]

        search = widgets.Text(
            value="",
            placeholder="Filter across visible and metadata fields...",
            description="Search:",
            layout=widgets.Layout(width="60%"),
        )

        sort_options = [c for c in columns if c in df.columns]
        sort_by = widgets.Dropdown(
            options=sort_options,
            value="name" if "name" in sort_options else sort_options[0],
            description="Sort by:",
            layout=widgets.Layout(width="260px"),
        )

        ascending = widgets.Checkbox(
            value=True,
            description="Ascending",
            layout=widgets.Layout(width="120px"),
        )

        limit = widgets.IntSlider(
            value=min(100, len(df)),
            min=1,
            max=max(1, len(df)),
            step=1,
            description="Rows:",
            continuous_update=False,
            layout=widgets.Layout(width="320px"),
        )

        column_selector = widgets.SelectMultiple(
            options=columns,
            value=tuple(columns),
            description="Columns:",
            layout=widgets.Layout(width="420px", height="180px"),
        )

        out = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #ccc",
                height="420px",
                overflow="auto",
                width="100%",
            )
        )

        summary = widgets.HTML()

        def apply_filters():
            filtered = df.copy()

            q = search.value.strip().lower()
            if q:
                mask = (
                    filtered[searchable_columns]
                    .fillna("")
                    .astype(str)
                    .apply(lambda col: col.str.lower().str.contains(q, regex=False))
                    .any(axis=1)
                )
                filtered = filtered[mask]

            if sort_by.value in filtered.columns:
                filtered = filtered.sort_values(
                    by=sort_by.value,
                    ascending=ascending.value,
                    kind="stable",
                )

            selected_columns = list(column_selector.value)
            if not selected_columns:
                selected_columns = ["name"]

            selected_columns = [c for c in selected_columns if c in filtered.columns]

            return filtered[selected_columns].head(limit.value)

        def render(_=None):
            filtered = apply_filters()
            with out:
                out.clear_output(wait=True)
                if filtered.empty:
                    display(HTML("<i>No matches.</i>"))
                else:
                    display(filtered.reset_index(drop=True))
            summary.value = f"<b>{len(filtered)}</b> row(s) shown"

        observers = [search, sort_by, ascending, limit, column_selector]
        for w in observers:
            w.observe(render, names="value")

        children = [
            search,
            widgets.HBox([sort_by, ascending, limit]),
            column_selector,
            summary,
            out,
        ]

        if include_source_inspector:
            class_fields = sorted(
                {
                    col[:-8]
                    for col in df.columns
                    if col.endswith("__object")
                }
            )

            if class_fields:
                inspect_field = widgets.Dropdown(
                    options=class_fields,
                    value=class_fields[0],
                    description="Inspect field:",
                    layout=widgets.Layout(width="320px"),
                )

                inspect_node = widgets.Dropdown(
                    options=sorted(df["name"].astype(str).tolist()),
                    description="Node:",
                    layout=widgets.Layout(width="420px"),
                )

                inspect_out = widgets.Output(
                    layout=widgets.Layout(
                        border="1px solid #ddd",
                        max_height="320px",
                        overflow="auto",
                        width="100%",
                    )
                )

                def render_source(_=None):
                    field = inspect_field.value
                    node_name = inspect_node.value

                    with inspect_out:
                        inspect_out.clear_output(wait=True)

                        row = df[df["name"].astype(str) == str(node_name)]
                        if row.empty:
                            display(HTML("<i>No node selected.</i>"))
                            return

                        obj_col = f"{field}__object"
                        if obj_col not in row.columns:
                            display(HTML(f"<i>No class object stored for field '{field}'.</i>"))
                            return

                        obj = row.iloc[0][obj_col]
                        if obj is None:
                            display(HTML("<i>Nothing to inspect.</i>"))
                            return

                        try:
                            src = inspect.getsource(obj)
                            src = src.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            display(HTML(f"<pre style='white-space: pre-wrap; margin: 0;'>{src}</pre>"))
                        except Exception as e:
                            display(HTML(f"<i>Could not load source: {e}</i>"))

                inspect_field.observe(render_source, names="value")
                inspect_node.observe(render_source, names="value")

                children.extend([
                    widgets.HTML("<hr>"),
                    widgets.HBox([inspect_field, inspect_node]),
                    inspect_out,
                ])

                render_source()

        ui = widgets.VBox(children)
        render()
        display(ui)
        return ui

    # --------------------------------------------------------
    # Public overview API
    # --------------------------------------------------------

    def list(
        self,
        *,
        notebook: bool | None = None,
        columns: list[str] | None = None,
        include_metadata: bool = False,
        include_source_inspector: bool = True,
    ):
        """Return a nicely formatted overview of the registry.

        In a notebook:
            shows an interactive widget if pandas + ipywidgets are available.

        Outside a notebook:
            returns a plain-text table string.

        Parameters
        ----------
        notebook:
            Force notebook mode on/off. If None, auto-detect.
        columns:
            Optional subset of columns to show.
        include_metadata:
            If True, also show generated metadata columns such as:
            node_class__module, node_class__doc, ...
        include_source_inspector:
            In notebook mode, show source viewer for class-like fields.

        """
        if notebook is None:
            notebook = _in_ipython_notebook()

        if notebook:
            try:
                return self._list_notebook(
                    columns=columns,
                    include_metadata=include_metadata,
                    include_source_inspector=include_source_inspector,
                )
            except ImportError:
                # hard fallback
                return self._plain_text_table(columns=columns)

        return self._plain_text_table(columns=columns)


NODE_REGISTRY = NodeRegistry()
