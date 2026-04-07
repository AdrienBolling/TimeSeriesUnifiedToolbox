"""Rendering mixin for Registry classes.

Provides plain-text and notebook (ipywidgets) display of registry contents.
Separated from the base :class:`Registry` so that CRUD logic stays lean and
the display surface can evolve independently.
"""

import inspect
from typing import Any, Protocol, runtime_checkable


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
    - scalars pass through
    - dicts are NOT collapsed here — they are unfolded by ``_rows`` instead
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

    return value


def _format_plain_value(value: Any) -> str:
    if isinstance(value, dict) and value.get("__kind__") == "class":
        return str(value.get("display", ""))
    return str(value)


def _clip(s: str, width: int) -> str:
    return s if len(s) <= width else s[: max(0, width - 1)] + "…"


# ============================================================
# Protocol — what the mixin expects from the host class
# ============================================================


@runtime_checkable
class _HasRegistry(Protocol):
    _registry: dict[str, dict[str, Any]]
    _entity: str
    _column_order: list[str]


# ============================================================
# Display mixin
# ============================================================


class RegistryDisplayMixin:
    """Mixin that adds rendering capabilities to a Registry.

    Expects the host class to have ``_registry`` (dict of entries) and
    ``_entity`` (str label like ``"node"``).
    """

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    def _rows(self: _HasRegistry) -> list[dict[str, Any]]:
        """Flatten registry entries into display rows.

        Dict-valued fields are *unfolded*: each key in the dict becomes its own
        column (prefixed with ``field.key``).  This makes metadata produced by
        e.g. ``model_dump()`` immediately visible without manual expansion.

        Entries that lack a particular sub-key get an empty string in that
        column, so the table stays rectangular.
        """
        rows: list[dict[str, Any]] = []

        for name, fields in self._registry.items():
            row: dict[str, Any] = {"name": name}

            for field_name, value in fields.items():
                # --- Dict unfolding ---
                if isinstance(value, dict) and not _is_class_like(value):
                    for sub_key, sub_value in value.items():
                        col_name = f"{field_name}.{sub_key}"
                        serialized = _serialize_value(sub_value)

                        if (
                            isinstance(serialized, dict)
                            and serialized.get("__kind__") == "class"
                        ):
                            row[col_name] = serialized["display"]
                            row[f"{col_name}__module"] = serialized["module"]
                            row[f"{col_name}__qualname"] = serialized["qualname"]
                            row[f"{col_name}__doc"] = serialized["doc"]
                            row[f"{col_name}__file"] = serialized["file"]
                            row[f"{col_name}__line"] = serialized["line"]
                            row[f"{col_name}__object"] = serialized["object"]
                        else:
                            row[col_name] = serialized
                    continue

                # --- Scalar / class / list ---
                serialized = _serialize_value(value)

                row[field_name] = (
                    serialized["display"]
                    if isinstance(serialized, dict)
                    and serialized.get("__kind__") == "class"
                    else serialized
                )

                if (
                    isinstance(serialized, dict)
                    and serialized.get("__kind__") == "class"
                ):
                    row[f"{field_name}__module"] = serialized["module"]
                    row[f"{field_name}__qualname"] = serialized["qualname"]
                    row[f"{field_name}__doc"] = serialized["doc"]
                    row[f"{field_name}__file"] = serialized["file"]
                    row[f"{field_name}__line"] = serialized["line"]
                    row[f"{field_name}__object"] = serialized["object"]

            rows.append(row)

        return rows

    def _display_columns(self: _HasRegistry) -> list[str]:
        """Select user-facing columns dynamically.

        If the host class defines ``_column_order`` (a list of column names),
        those columns appear first in that exact order.  Remaining columns
        follow alphabetically, with internal ``__``-suffixed metadata columns
        placed last.
        """
        rows = self._rows()
        if not rows:
            return ["name"]

        all_columns: set[str] = set()
        for row in rows:
            all_columns.update(row.keys())

        # Forced order from the host class (if defined and non-empty).
        forced = list(getattr(self, "_column_order", None) or [])
        ordered = [c for c in forced if c in all_columns]
        seen = set(ordered)

        # Remaining direct fields (no __ suffix), alphabetically.
        direct_remaining = sorted(
            c for c in all_columns if c not in seen and "__" not in c
        )
        ordered.extend(direct_remaining)
        seen.update(direct_remaining)

        # Metadata columns last.
        metadata_fields = sorted(
            c
            for c in all_columns
            if c not in seen and "__" in c and not c.endswith("__object")
        )
        ordered.extend(metadata_fields)

        return ordered

    # --------------------------------------------------------
    # Plain text rendering
    # --------------------------------------------------------

    def _plain_text_table(
        self: _HasRegistry,
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

        widths: dict[str, int] = {}
        for col in columns:
            widths[col] = min(
                max(len(col), max(len(r[col]) for r in display_rows)),
                max_col_width,
            )

        total_width = sum(widths.values()) + 3 * (len(columns) - 1)
        if total_width > max_width:
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
                " | ".join(
                    _clip(row[col], widths[col]).ljust(widths[col])
                    for col in columns
                )
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # Notebook rendering
    # --------------------------------------------------------

    def _list_notebook(
        self: _HasRegistry,
        columns: list[str] | None = None,
        include_metadata: bool = False,
        include_source_inspector: bool = True,
    ):
        import ipywidgets as widgets
        import pandas as pd
        from IPython.display import display

        rows = self._rows()
        if not rows:
            display(widgets.HTML("<b>Registry is empty.</b>"))
            return

        df = pd.DataFrame(rows)

        if columns is None:
            columns = self._display_columns()

        if not include_metadata:
            columns = [c for c in columns if "__" not in c]

        columns = [c for c in columns if c in df.columns]
        if "name" not in columns:
            columns = ["name"] + columns

        searchable_columns = [
            c for c in df.columns if not c.endswith("__object")
        ]

        # --- Control widgets -------------------------------------------------

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

        # Use widgets.HTML instead of widgets.Output so the browser
        # handles overflow natively — Output widgets do not reliably
        # contain their children or support horizontal scrolling.
        table_html = widgets.HTML(value="")
        summary = widgets.HTML()

        def apply_filters():
            filtered = df.copy()

            q = search.value.strip().lower()
            if q:
                mask = (
                    filtered[searchable_columns]
                    .fillna("")
                    .astype(str)
                    .apply(
                        lambda col: col.str.lower().str.contains(
                            q, regex=False
                        )
                    )
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

            selected_columns = [
                c for c in selected_columns if c in filtered.columns
            ]

            return filtered[selected_columns].head(limit.value)

        def render(_=None):
            filtered = apply_filters()
            if filtered.empty:
                table_html.value = "<i>No matches.</i>"
            else:
                raw_html = filtered.reset_index(drop=True).to_html()
                table_html.value = (
                    "<div style='"
                    "overflow-x: auto; "
                    "overflow-y: auto; "
                    "max-height: 420px; "
                    "border: 1px solid #ccc; "
                    "width: 100%;'>"
                    f"{raw_html}"
                    "</div>"
                )
            summary.value = f"<b>{len(filtered)}</b> row(s) shown"

        observers = [search, sort_by, ascending, limit, column_selector]
        for w in observers:
            w.observe(render, names="value")

        children: list[widgets.Widget] = [
            search,
            widgets.HBox([sort_by, ascending, limit]),
            column_selector,
            summary,
            table_html,
        ]

        # --- Source inspector ------------------------------------------------

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

                source_html = widgets.HTML(value="")

                def render_source(_=None):
                    field = inspect_field.value
                    node_name = inspect_node.value

                    matched = df[df["name"].astype(str) == str(node_name)]
                    if matched.empty:
                        source_html.value = (
                            "<i>No node selected.</i>"
                        )
                        return

                    obj_col = f"{field}__object"
                    if obj_col not in matched.columns:
                        source_html.value = (
                            "<i>Field not available for this node.</i>"
                        )
                        return

                    obj = matched.iloc[0][obj_col]
                    if obj is None or not _is_class_like(obj):
                        source_html.value = (
                            "<i>Field not available for this node.</i>"
                        )
                        return

                    try:
                        src = inspect.getsource(obj)
                        src = (
                            src.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )
                        source_html.value = (
                            "<div style='"
                            "max-height: 320px; "
                            "overflow: auto; "
                            "border: 1px solid #ddd; "
                            "width: 100%;'>"
                            f"<pre style='white-space: pre-wrap; margin: 0;'>{src}</pre>"
                            "</div>"
                        )
                    except Exception:
                        source_html.value = (
                            "<i>Field not available for this node.</i>"
                        )

                inspect_field.observe(render_source, names="value")
                inspect_node.observe(render_source, names="value")

                children.extend(
                    [
                        widgets.HTML("<hr>"),
                        widgets.HBox([inspect_field, inspect_node]),
                        source_html,
                    ]
                )

                render_source()

        ui = widgets.VBox(children)
        render()
        display(ui)

    # --------------------------------------------------------
    # Public overview API
    # --------------------------------------------------------

    def list(
        self: _HasRegistry,
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
                self._list_notebook(
                    columns=columns,
                    include_metadata=include_metadata,
                    include_source_inspector=include_source_inspector,
                )
                return None
            except ImportError:
                return self._plain_text_table(columns=columns)

        return self._plain_text_table(columns=columns)
