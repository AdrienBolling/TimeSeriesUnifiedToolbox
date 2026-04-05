from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from tsut.core.common.enums import NodeExecutionMode
from tsut.core.nodes.node import NodeConfig, NodeType

if TYPE_CHECKING:
    from tsut.core.pipeline.pipeline import Edge, Pipeline


# =============================================================================
# Graph structure helpers
# =============================================================================


def split_execution_graph_into_columns(
    pipeline: Pipeline,
) -> dict[int, set[str]]:
    """Group nodes into layout columns for visualization.

    The main execution graph is laid out from left to right using shortest-path
    distance to the sink node. Metric nodes are forced into a dedicated column.
    """
    graph = pipeline.graph
    main_graph = pipeline.graph_wo_metrics
    sink_node = pipeline.sink_node_name
    node_columns: dict[str, int | None] = dict.fromkeys(main_graph.nodes)

    shortest_paths_to_sink = nx.shortest_path_length(main_graph, target=sink_node)
    max_depth = max(shortest_paths_to_sink.values())

    for node, depth in shortest_paths_to_sink.items():
        if node is not None:
            node_columns[node] = -int(depth)

    # Force all source nodes to the far left.
    source_node_names = pipeline.get_source_node_names()
    if source_node_names:
        for source_node in source_node_names:
            node_columns[source_node] = -int(max_depth) - 1

    # Place metric nodes in their own column.
    for node in graph.nodes:
        if pipeline.internal_config.nodes[node][1].node_type == NodeType.METRIC:
            node_columns[node] = 1

    columns: dict[int, set[str]] = {}
    for node, col in node_columns.items():
        if col is not None:
            columns.setdefault(col, set()).add(node)

    return columns


# =============================================================================
# Node mappings
# =============================================================================


def node_name_to_color_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, str]:
    """Map node names to colors based on node type."""
    type_to_color = {
        NodeType.SOURCE: "lightblue",
        NodeType.SINK: "lightcoral",
        NodeType.METRIC: "lightgreen",
        NodeType.TRANSFORM: "orange",
        NodeType.MODEL: "lightgray",
    }
    return {
        node_name: type_to_color.get(node_config.node_type, "white")
        for node_name, (_, node_config) in node_configs.items()
    }


def chosen_data_from_node(node_config: NodeConfig) -> Any:
    """Extract the data displayed for a node in hover/tooltip views."""
    return node_config.model_dump()


def node_name_to_node_data_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, dict[str, Any]]:
    """Map node names to serialized node data."""
    return {
        node_name: chosen_data_from_node(node_config)
        for node_name, (_, node_config) in node_configs.items()
    }


def node_name_to_marker_style_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, str]:
    """Map node names to matplotlib marker styles based on node type."""
    type_to_marker = {
        NodeType.SOURCE: "p",
        NodeType.SINK: "p",
        NodeType.METRIC: "d",
        NodeType.TRANSFORM: "^",
        NodeType.MODEL: "o",
    }
    return {
        node_name: type_to_marker.get(node_config.node_type, "x")
        for node_name, (_, node_config) in node_configs.items()
    }


def node_name_to_hover_text_mapping(
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[str, str]:
    """Map node names to plain-text hover content."""
    hover_text_mapping: dict[str, str] = {}
    node_data_mapping = node_name_to_node_data_mapping(node_configs)

    for node_name, node_data in node_data_mapping.items():
        meta_dict = (
            node_data if isinstance(node_data, dict) else {"value": str(node_data)}
        )
        lines = [f"node: {node_name}"]

        if meta_dict:
            for key, value in meta_dict.items():
                lines.append(f"{key}: {value}")
        else:
            lines.append("(no metadata)")

        hover_text_mapping[node_name] = "\n".join(lines)

    return hover_text_mapping


# =============================================================================
# Edge mappings
# =============================================================================


def get_execution_mode(
    node_config: NodeConfig, port_name: str, source: bool
) -> list[str]:
    """Return the execution modes for a port.

    Parameters
    ----------
    source:
        True when reading an input port, False when reading an output port.

    """
    if source:
        return node_config.in_ports[port_name].mode
    return node_config.out_ports[port_name].mode


def edge_to_linestyle_mapping(
    edges: list[Edge],
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[tuple[str, str], str]:
    """Map edges to line styles based on connected node types."""
    mapping: dict[tuple[str, str], str] = {}

    for edge in edges:
        source_type = node_configs[edge.source][1].node_type
        target_type = node_configs[edge.target][1].node_type

        if source_type == NodeType.SOURCE or target_type == NodeType.SINK:
            mapping[(edge.source, edge.target)] = "-"
        elif source_type == NodeType.METRIC or target_type == NodeType.METRIC:
            mapping[(edge.source, edge.target)] = "--"
        else:
            mapping[(edge.source, edge.target)] = "-"

    return mapping


def select_edges_to_display(
    edges: list[Edge],
    node_configs: dict[str, tuple[str, NodeConfig]],
    execution_mode: str | None = None,
) -> list[Edge]:
    """Return the subset of edges to display.

    Parameters
    ----------
    execution_mode:
        - None: display all edges
        - any mode name: display only edges matching that execution mode

    Notes
    -----
    This function assumes you already have a helper such as
    `edge_matches_execution_mode(edge, execution_mode)` available elsewhere.

    """
    if execution_mode is None or execution_mode == NodeExecutionMode.ALL:
        return edges

    source_configs = {node: config for node, (_, config) in node_configs.items()}
    target_configs = {node: config for node, (_, config) in node_configs.items()}

    return [
        edge
        for edge in edges
        if edge_matches_execution_mode(
            edge,
            execution_mode,
            source_configs[edge.source],
            target_configs[edge.target],
        )
    ]


def edge_matches_execution_mode(
    edge: Edge,
    execution_mode: str,
    source_config: NodeConfig,
    target_config: NodeConfig,
) -> bool:
    """Return True if the edge has at least one port_map matching the given execution mode."""
    exec_modes = [
        (
            source_config.out_ports[source_port].mode,
            target_config.in_ports[target_port].mode,
        )
        for (source_port, target_port) in edge.ports_map
    ]
    for source_modes, target_modes in exec_modes:
        if "all" in source_modes and "all" in target_modes:
            return True
        if execution_mode in source_modes or execution_mode in target_modes:
            return True
    return False


def edge_to_hover_text_mapping(edges: list[Edge]) -> dict[tuple[str, str], str]:
    """Map edges to plain-text hover content."""
    mapping: dict[tuple[str, str], str] = {}

    for edge in edges:
        lines = [f"edge: {edge.source} -> {edge.target}", "Ports mapping:"]
        lines.extend(
            f"  {target_port}: {source_port}"
            for (source_port, target_port) in edge.ports_map
        )
        mapping[(edge.source, edge.target)] = "\n".join(lines)

    return mapping


def edge_curved_mapping(
    edges: list[Edge],
    node_configs: dict[str, tuple[str, NodeConfig]],
) -> dict[tuple[str, str], bool]:
    """Map edges to whether they should be rendered as curved.

    Edges targeting metric nodes are curved to visually separate them from the
    main execution flow.
    """
    return {
        (edge.source, edge.target): node_configs[edge.target][1].node_type
        == NodeType.METRIC
        for edge in edges
    }


# =============================================================================
# Hover HTML helpers for Plotly
# =============================================================================


def _dict_to_hover_html(data: Any, indent: int = 0) -> str:
    """Format nested dict/list data as HTML for Plotly hover content."""
    pad = "&nbsp;" * 4 * indent

    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            escaped_key = html.escape(str(key))
            if isinstance(value, (dict, list, tuple)):
                lines.append(f"{pad}<b>{escaped_key}</b>:")
                lines.append(_dict_to_hover_html(value, indent + 1))
            else:
                lines.append(f"{pad}<b>{escaped_key}</b>: {html.escape(str(value))}")
        return "<br>".join(lines)

    if isinstance(data, (list, tuple)):
        lines = []
        for item in data:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{pad}•")
                lines.append(_dict_to_hover_html(item, indent + 1))
            else:
                lines.append(f"{pad}• {html.escape(str(item))}")
        return "<br>".join(lines)

    return f"{pad}{html.escape(str(data))}"


def _node_hover_html(node_name: str, node_data: Any) -> str:
    title = f"<b>{html.escape(str(node_name))}</b>"
    body = _dict_to_hover_html(node_data)
    return f"{title}<br>{body}" if body else title


def _edge_hover_html(source: str, target: str, ports_map: list[tuple[str, str]]) -> str:
    lines = [
        f"<b>{html.escape(source)} → {html.escape(target)}</b>",
        "<b>Ports mapping</b>",
    ]
    lines.extend(
        f"{html.escape(str(source_port))} → {html.escape(str(target_port))}"
        for source_port, target_port in ports_map
    )
    return "<br>".join(lines)


# =============================================================================
# Geometry helpers for curved edges
# =============================================================================


def _vertical_center(pos: dict[str, Any]) -> float:
    ys = [float(coords[1]) for coords in pos.values()]
    return (min(ys) + max(ys)) / 2.0


def _edge_curvature(
    source: str,
    pos: dict[str, Any],
    y_center: float,
    min_curvature: float = 0.18,
    max_curvature: float = 0.45,
    power: float = 1.2,
) -> float:
    """Compute signed curvature for a curved edge from source position.

    Edges above the layout center bend upward, edges below bend downward.
    The farther the source from the center, the stronger the curvature.
    """
    source_y = float(pos[source][1])
    ys = [float(coords[1]) for coords in pos.values()]
    max_dist = max(*(abs(y - y_center) for y in ys), 1e-12)

    norm_dist = min(max(abs(source_y - y_center) / max_dist, 0.0), 1.0)
    magnitude = min_curvature + (max_curvature - min_curvature) * (norm_dist**power)
    sign = 1.0 if source_y >= y_center else -1.0
    return sign * magnitude


def _quadratic_bezier_control_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
) -> tuple[float, float]:
    """Return the control point for a quadratic Bézier curve."""
    x0, y0 = p0
    x1, y1 = p1

    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy) or 1.0

    nx_, ny_ = -dy / dist, dx / dist
    return mx + curvature * dist * nx_, my + curvature * dist * ny_


def _quadratic_bezier_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    control: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate a quadratic Bézier curve at parameter t."""
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = control

    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
    return x, y


def _quadratic_bezier_derivative(
    p0: tuple[float, float],
    p1: tuple[float, float],
    control: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate the derivative of a quadratic Bézier curve at parameter t."""
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = control

    dx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
    dy = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)
    return dx, dy


def _quadratic_bezier_polyline(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
    steps: int = 200,
) -> tuple[list[float], list[float], tuple[float, float]]:
    """Sample a quadratic Bézier curve into x/y coordinate lists."""
    control = _quadratic_bezier_control_point(p0, p1, curvature)
    ts = np.linspace(0.0, 1.0, steps + 1)

    xs: list[float] = []
    ys: list[float] = []
    for t in ts:
        x, y = _quadratic_bezier_point(p0, p1, control, float(t))
        xs.append(x)
        ys.append(y)

    return xs, ys, control


def _bezier_t_for_arc_fraction(
    p0: tuple[float, float],
    p1: tuple[float, float],
    control: tuple[float, float],
    steps: int = 400,
    arc_fraction: float = 0.5,
) -> float:
    """Approximate the parameter t corresponding to a target arc-length fraction."""
    ts = np.linspace(0.0, 1.0, steps + 1)
    points = np.array([_quadratic_bezier_point(p0, p1, control, float(t)) for t in ts])

    seg_lengths = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]

    if total == 0:
        return 0.5

    target = arc_fraction * total
    idx = int(np.searchsorted(cum_lengths, target))
    idx = max(1, min(idx, len(ts) - 1))

    prev_len = cum_lengths[idx - 1]
    next_len = cum_lengths[idx]
    alpha = 0.0 if next_len == prev_len else (target - prev_len) / (next_len - prev_len)
    return float(ts[idx - 1] + alpha * (ts[idx] - ts[idx - 1]))


def _sample_quadratic_bezier_with_exact_label_pose(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
    steps: int = 200,
    label_arc_fraction: float = 0.5,
) -> tuple[list[float], list[float], float, float, float]:
    """Sample a Bézier curve and compute an exact label position and angle."""
    xs, ys, control = _quadratic_bezier_polyline(
        p0=p0,
        p1=p1,
        curvature=curvature,
        steps=steps,
    )

    t_label = _bezier_t_for_arc_fraction(
        p0=p0,
        p1=p1,
        control=control,
        steps=max(400, steps),
        arc_fraction=label_arc_fraction,
    )

    lx, ly = _quadratic_bezier_point(p0, p1, control, t_label)
    dx, dy = _quadratic_bezier_derivative(p0, p1, control, t_label)

    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return xs, ys, lx, ly, angle


# =============================================================================
# Plotly rendering helpers
# =============================================================================


def _add_edge_shapes(
    fig: go.Figure,
    edges: list[Edge],
    pos: dict[str, Any],
    edge_color_mapping: dict[tuple[str, str], str],
    edge_linestyle_mapping: dict[tuple[str, str], str],
    edge_curved_mapping_: dict[tuple[str, str], bool],
) -> None:
    """Add visible straight edge shapes to a Plotly figure.

    Hover is handled separately with transparent scatter traces.
    """
    dash_map = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}

    for edge in edges:
        source, target = edge.source, edge.target
        if edge_curved_mapping_.get((source, target), False):
            continue

        x0, y0 = float(pos[source][0]), float(pos[source][1])
        x1, y1 = float(pos[target][0]), float(pos[target][1])

        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line={
                "color": edge_color_mapping.get((source, target), "gray"),
                "width": 2,
                "dash": dash_map.get(
                    edge_linestyle_mapping.get((source, target), "-"), "solid"
                ),
            },
            layer="below",
        )


def _add_legend(
    fig: go.Figure,
    graph: nx.Graph,
    node_configs: dict[str, tuple[str, NodeConfig]],
    node_color_mapping: dict[str, str],
    node_marker_mapping: dict[str, str],
    edge_color_mapping: dict[tuple[str, str], str],
) -> None:
    """Add node and edge legend entries for the styles present in the graph."""
    marker_symbol_map = {
        "o": "circle",
        "s": "square",
        "^": "triangle-up",
        "v": "triangle-down",
        "d": "diamond",
        "p": "pentagon",
        "x": "x",
        "*": "star",
    }

    seen_node_types: dict[NodeType, str] = {}
    for node in graph.nodes:
        node_type = node_configs[node][1].node_type
        if node_type in seen_node_types:
            continue
        seen_node_types[node_type] = node

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": node_color_mapping[node],
                    "symbol": marker_symbol_map.get(
                        node_marker_mapping[node], "circle"
                    ),
                    "line": {"color": "black", "width": 1},
                },
                name=node_type.name,
                legendgroup="nodes",
                showlegend=True,
            )
        )

    color_to_label = {
        "black": "All modes",
        "blue": "Multiple modes",
        "green": "Train",
        "orange": "Inference",
        "purple": "Evaluation",
        "gray": "Other",
    }

    seen_colors: set[str] = set()
    for color in edge_color_mapping.values():
        if color in seen_colors:
            continue
        seen_colors.add(color)

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": color, "width": 3},
                name=color_to_label.get(color, color),
                legendgroup="edges",
                showlegend=True,
            )
        )


def edge_to_color_mapping(
    edges: list[Edge],
    execution_mode: str | None = None,
) -> dict[tuple[str, str], str]:
    """Map edges to colors based on execution mode."""
    if execution_mode is None:
        execution_mode = str(NodeExecutionMode.ALL)
    exec_to_color_mapping = {
        NodeExecutionMode.ALL: "black",
        NodeExecutionMode.TRAINING: "green",
        NodeExecutionMode.INFERENCE: "orange",
        NodeExecutionMode.EVALUATION: "purple",
    }
    return {
        (edge.source, edge.target): exec_to_color_mapping[
            NodeExecutionMode(execution_mode)
        ]
        for edge in edges
    }


def _straight_edge_polyline(
    p0: tuple[float, float],
    p1: tuple[float, float],
    steps: int = 100,
) -> tuple[list[float], list[float]]:
    """Sample a straight edge into many points for reliable hover along the line."""
    x0, y0 = p0
    x1, y1 = p1
    ts = np.linspace(0.0, 1.0, steps + 1)

    xs = [float((1 - t) * x0 + t * x1) for t in ts]
    ys = [float((1 - t) * y0 + t * y1) for t in ts]
    return xs, ys


# =============================================================================
# Plotly rendering
# =============================================================================


def render_pipeline_graph_plotly(
    pipeline: Pipeline,
    title: str = "Pipeline Graph",
    figsize: tuple[int, int] = (12, 6),
    execution_mode: str | None = None,
) -> go.Figure:
    """Render the pipeline graph with Plotly and a dropdown to select execution mode."""
    version = pipeline.version

    graph = pipeline.graph
    node_configs = pipeline.internal_config.nodes

    columns = split_execution_graph_into_columns(pipeline=pipeline)
    pos = nx.multipartite_layout(graph, subset_key=columns)
    y_center = _vertical_center(pos)

    node_color_mapping = node_name_to_color_mapping(node_configs)
    node_marker_mapping = node_name_to_marker_style_mapping(node_configs)
    node_data_mapping = node_name_to_node_data_mapping(node_configs)

    marker_symbol_map = {
        "o": "circle",
        "s": "square",
        "^": "triangle-up",
        "v": "triangle-down",
        "d": "diamond",
        "p": "pentagon",
        "x": "x",
        "*": "star",
    }
    dash_map = {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}

    fig = go.Figure()

    # -------------------------------------------------------------------------
    # Build edge traces for each execution mode
    # -------------------------------------------------------------------------
    mode_order = [
        str(NodeExecutionMode.ALL),
        str(NodeExecutionMode.TRAINING),
        str(NodeExecutionMode.INFERENCE),
        str(NodeExecutionMode.EVALUATION),
    ]

    if execution_mode is None:
        initial_mode = str(NodeExecutionMode.ALL)
    else:
        initial_mode = str(NodeExecutionMode(execution_mode))

    edge_trace_indices_by_mode: dict[str, list[int]] = {mode: [] for mode in mode_order}

    for mode in mode_order:
        visible_edges = select_edges_to_display(
            pipeline.edges,
            execution_mode=mode,
            node_configs=node_configs,
        )
        edge_color_mapping = edge_to_color_mapping(visible_edges, mode)
        edge_linestyle_mapping = edge_to_linestyle_mapping(visible_edges, node_configs)
        edge_curved_mapping_ = edge_curved_mapping(visible_edges, node_configs)

        for edge in visible_edges:
            source, target = edge.source, edge.target
            x0, y0 = float(pos[source][0]), float(pos[source][1])
            x1, y1 = float(pos[target][0]), float(pos[target][1])
            curved = edge_curved_mapping_.get((source, target), False)

            if curved:
                curvature = _edge_curvature(
                    source=source,
                    pos=pos,
                    y_center=y_center,
                    min_curvature=0.10,
                    max_curvature=0.45,
                    power=1.2,
                )

                xs, ys, _, _, _ = _sample_quadratic_bezier_with_exact_label_pose(
                    (x0, y0),
                    (x1, y1),
                    curvature=curvature,
                    steps=200,
                    label_arc_fraction=0.5,
                )
            else:
                xs, ys = _straight_edge_polyline(
                    (x0, y0),
                    (x1, y1),
                    steps=100,
                )

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={
                        "color": edge_color_mapping.get((source, target), "gray"),
                        "width": 2,
                        "dash": dash_map.get(
                            edge_linestyle_mapping.get((source, target), "-"),
                            "solid",
                        ),
                    },
                    hovertemplate=_edge_hover_html(source, target, edge.ports_map)
                    + "<extra></extra>",
                    showlegend=False,
                    visible=(mode == initial_mode),
                )
            )
            edge_trace_indices_by_mode[mode].append(len(fig.data) - 1)

    # -------------------------------------------------------------------------
    # Node trace
    # -------------------------------------------------------------------------
    nodes_in_order = list(graph.nodes)
    node_x = [float(pos[node][0]) for node in nodes_in_order]
    node_y = [float(pos[node][1]) for node in nodes_in_order]
    node_hover = [
        _node_hover_html(node, node_data_mapping.get(node, {}))
        for node in nodes_in_order
    ]
    node_symbols = [
        marker_symbol_map.get(node_marker_mapping.get(node, "o"), "circle")
        for node in nodes_in_order
    ]
    node_colors = [node_color_mapping.get(node, "white") for node in nodes_in_order]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(node) for node in nodes_in_order],
            textposition="top center",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=node_hover,
            marker={
                "size": 28,
                "color": node_colors,
                "symbol": node_symbols,
                "line": {"color": "black", "width": 1},
            },
            textfont={"size": 11, "color": "black"},
            showlegend=False,
            visible=True,
        )
    )
    node_trace_index = len(fig.data) - 1

    # -------------------------------------------------------------------------
    # Legend entries
    # Keep them always visible
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 0},
            name="— Nodes —",
            showlegend=True,
            visible=True,
        )
    )
    separator_trace_index = len(fig.data) - 1

    # Node legend
    seen_node_types: dict[NodeType, str] = {}
    node_legend_trace_indices: list[int] = []

    for node in graph.nodes:
        node_type = node_configs[node][1].node_type
        if node_type in seen_node_types:
            continue
        seen_node_types[node_type] = node

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 12,
                    "color": node_color_mapping[node],
                    "symbol": marker_symbol_map.get(
                        node_marker_mapping[node], "circle"
                    ),
                    "line": {"color": "black", "width": 1},
                },
                name=node_type.name,
                legendgroup="nodes",
                showlegend=True,
                visible=True,
            )
        )
        node_legend_trace_indices.append(len(fig.data) - 1)

    # Edge legend
    color_to_label = {
        "black": "All modes",
        "green": "Training",
        "orange": "Inference",
        "purple": "Evaluation",
        "gray": "Other",
    }

    edge_legend_trace_indices: list[int] = []
    for color in ["black", "green", "orange", "purple"]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": color, "width": 3},
                name=color_to_label[color],
                legendgroup="edges",
                showlegend=True,
                visible=True,
            )
        )
        edge_legend_trace_indices.append(len(fig.data) - 1)

    # -------------------------------------------------------------------------
    # Dropdown visibility masks
    # -------------------------------------------------------------------------
    always_visible = {
        node_trace_index,
        separator_trace_index,
        *node_legend_trace_indices,
        *edge_legend_trace_indices,
    }

    buttons = []
    total_traces = len(fig.data)

    for mode in mode_order:
        visible_mask = [False] * total_traces

        for idx in always_visible:
            visible_mask[idx] = True

        for idx in edge_trace_indices_by_mode[mode]:
            visible_mask[idx] = True

        label = mode.capitalize()

        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible_mask},
                    {
                        "title": {
                            "text": (
                                f"{title} — {label}"
                                f"<br><span style='font-size:12px;color:gray'>{version!s}</span>"
                            ),
                            "x": 0.5,
                            "xanchor": "center",
                        }
                    },
                ],
            }
        )

    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    fig.update_layout(
        title={
            "text": (
                f"{title} — {initial_mode.capitalize()}"
                f"<br><span style='font-size:12px;color:gray'>{version!s}</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel={
            "bgcolor": "rgba(30,30,30,0.95)",
            "font": {"color": "white", "size": 12},
            "align": "left",
        },
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        legend={
            "title": "Legend",
            "orientation": "v",
            "x": 1.02,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
            "bordercolor": "rgba(0,0,0,0.1)",
            "borderwidth": 1,
        },
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.01,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
        ],
    )

    return fig
