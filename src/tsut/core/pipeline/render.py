from typing import TYPE_CHECKING, Any

import iplotx as ipx
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx

from tsut.core.common.version import Version
from tsut.core.nodes.node import Node, NodeConfig, NodeType

if TYPE_CHECKING:
    from tsut.core.pipeline.pipeline import Edge, Pipeline

import html
import math

import plotly.graph_objects as go


def split_execution_graph_into_columns(
        graph: nx.Graph,
        graph_wo_metrics: nx.Graph,
        source_node: str,
        sink_node: str
) -> list[set[str]]:
    """Split the execution graph of the pipeline into columns for visualization.a"""
    # Get the graph from the pipeline, without the metric nodes for now since they will be placed in the bottom row
    gr = graph_wo_metrics
    # Initialize a dictionary to hold the column index for each node
    node_columns = dict.fromkeys(gr.nodes)
    # First, assign columns to source and sink nodes
    source = source_node
    sink = sink_node

    node_columns[source] = 0
    node_columns[sink] = -1  # Temporarily assign sink to column -1

    # Find the shortest path from each node to the sink to determine its depth
    shortest_paths_to_sink = nx.shortest_path_length(gr, target=sink)

    # Now we know the depth of each node, we can assign columns
    # The sink node should be in the right-most column, which is max depth + 1
    max_depth = max(shortest_paths_to_sink.values())
    for node, depth in shortest_paths_to_sink.items():
        node_columns[node] = -depth  # Assign columns based on depth, with sink at max_depth

    # Place all metric nodes in the last column (column index max_depth + 1)
    full_graph = graph
    for node in full_graph.nodes:
        if node in gr.nodes:
            continue  # Skip nodes already placed
        # These are metric nodes, place them in the last column
        node_columns[node] = 1

    # Ensure source is in the left-most column (column index 0)
    node_columns[source] = -(max_depth + 1)

    # Group nodes by their assigned column
    columns = {}
    for node, col in node_columns.items():
        if col is not None:
            columns.setdefault(col, set()).add(node)

    return columns

def node_name_to_color_mapping(node_configs: dict[str, tuple[type[Node], NodeConfig]]) -> dict[str, str]:
    """Create a mapping from node names to colors based on their type."""
    color_mapping = {}
    for node_name, (_, node_config) in node_configs.items():
        if node_config.node_type == NodeType.SOURCE:
            color_mapping[node_name] = "lightblue"
        elif node_config.node_type == NodeType.SINK:
            color_mapping[node_name] = "lightcoral"
        elif node_config.node_type == NodeType.METRIC:
            color_mapping[node_name] = "lightgreen"
        elif node_config.node_type == NodeType.TRANSFORM:
            color_mapping[node_name] = "orange"
        elif node_config.node_type == NodeType.MODEL:
            color_mapping[node_name] = "lightgray"
        else:
            color_mapping[node_name] = "white"  # Default color for unknown types
    return color_mapping

def chosen_data_from_node(node_config: NodeConfig) -> Any:
    """Extract the chosen data from a node configuration for display in the visualization."""
    # This function can be customized based on the specific attributes of your Node objects
    # For example, you might want to display the node's parameters, output shape, or other relevant information
    return node_config.model_dump()  # Example: return the node's configuration as a dictionary

def node_name_to_node_data_mapping(node_configs: dict[str, tuple[type[Node], NodeConfig]]) -> dict[str, dict[str, Any]]:
    """Create a mapping from node names to their instantiated Node objects."""
    node_mapping = {}
    for node_name, (_, node_config) in node_configs.items():
        # Instantiate the node and extract its data
        node_mapping[node_name] = chosen_data_from_node(node_config)
    return node_mapping

def node_name_to_marker_style_mapping(node_configs: dict[str, tuple[type[Node], NodeConfig]]) -> dict[str, str]:
    """Create a mapping from node names to marker styles based on their type."""
    marker_mapping = {}
    for node_name, (_, node_config) in node_configs.items():
        # Instantiate the node
        if node_config.node_type == NodeType.SOURCE:
            marker_mapping[node_name] = "p"  # Circle for source nodes
        elif node_config.node_type == NodeType.SINK:
            marker_mapping[node_name] = "p"  # Square for sink nodes
        elif node_config.node_type == NodeType.METRIC:
            marker_mapping[node_name] = "d"  # Diamond for metric nodes
        elif node_config.node_type == NodeType.TRANSFORM:
            marker_mapping[node_name] = "^"  # Triangle for transform nodes
        elif node_config.node_type == NodeType.MODEL:
            marker_mapping[node_name] = "o"  # Inverted triangle for model nodes
        else:
            marker_mapping[node_name] = "x"  # Cross for unknown types
    return marker_mapping

def node_name_to_hover_text_mapping(node_configs: dict[str, tuple[type[Node], NodeConfig]]) -> dict[str, str]:
    """Create a mapping from node names to hover text that includes metadata information."""
    hover_text_mapping = {}
    dict_mapping = node_name_to_node_data_mapping(node_configs)
    for node_name, node_data in dict_mapping.items():
        if isinstance(node_data, dict):
            meta_dict = node_data
        else:
            meta_dict = {"value": str(node_data)}
        lines = [f"node: {node_name}"]
        if meta_dict:
            for k, v in meta_dict.items():
                lines.append(f"{k}: {v}")
        else:
            lines.append("(no metadata)")

        hover_text_mapping[node_name] = "\n".join(lines)
    return hover_text_mapping

def edge_to_linestyle_mapping(edges: list["Edge"], node_configs: dict[str, tuple[type[Node], NodeConfig]]) -> dict[tuple[str, str], str]:
    """Create a mapping from edges (source, target) to line styles based on the types of the connected nodes."""
    linestyle_mapping = {}
    for edge in edges:
        source, target = edge.source, edge.target
        source_type = node_configs[source][1].node_type
        target_type = node_configs[target][1].node_type
        if source_type == NodeType.SOURCE or target_type == NodeType.SINK:
            linestyle_mapping[(source, target)] = "-"  # Edges connected to source or sink are solid
        elif source_type == NodeType.METRIC or target_type == NodeType.METRIC:
            linestyle_mapping[(source, target)] = "--"
        else:
            linestyle_mapping[(source, target)] = "-"  # Other edges are solid
    return linestyle_mapping

def get_execution_mode(node_config: NodeConfig, port_name: str, source: bool) -> list[str]:
    """Get the execution mode of a port from the node configuration."""
    # This function can be customized based on how you define execution modes in your NodeConfig
    # For example, you might have an attribute in NodeConfig that specifies the execution mode for each port
    if source:
        return node_config.in_ports[port_name].mode
    return node_config.out_ports[port_name].mode

def edge_to_color_mapping(edges: list["Edge"], node_configs: dict[str, tuple[type[Node], NodeConfig]]) -> dict[tuple[str, str], str]:
    """Create a mapping from edges to colours based on the execution mode of the connected ports.

    Case :
    - "all" mode: black line
    - several but not all modes: blue line
    - "train" only: green line
    - "inference" only: orange line
    - "evaluation" only: purple line
    """
    mapping = {}
    for edge in edges:
        source, target, ports_map = edge.source, edge.target, edge.ports_map
        target_port, source_port = list(ports_map.items())[0]
        source_node_config = node_configs[source][1]
        target_node_config = node_configs[target][1]
        source_modes = get_execution_mode(source_node_config, source_port, source=False)
        target_modes = get_execution_mode(target_node_config, target_port, source=True)
        # For simplicity, we assume that the execution mode of the edge is determined by the intersection of the modes of the connected ports
        edge_modes = set(source_modes).intersection(set(target_modes))
        if "all" in edge_modes:
            mapping[(source, target)] = "black"
        elif len(edge_modes) > 1:
            mapping[(source, target)] = "blue"
        elif "train" in edge_modes:
            mapping[(source, target)] = "green"
        elif "inference" in edge_modes:
            mapping[(source, target)] = "orange"
        elif "evaluation" in edge_modes:
            mapping[(source, target)] = "purple"
        else:
            mapping[(source, target)] = "gray"  # Default color for edges with no specific mode
    return mapping


def edge_to_hover_text_mapping(edges: list["Edge"]) -> dict[tuple[str, str], str]:
    """Create a mapping from edges (source, target) to hover text that includes metadata information."""
    hover_text_mapping = {}
    for edge in edges:
        source, target, mapping = edge.source, edge.target, edge.ports_map
        edge_info = f"edge: {source} -> {target}\nPorts mapping:\n"
        for k, v in mapping.items():
            edge_info += f"  {k}: {v}\n"
        hover_text_mapping[(source, target)] = edge_info
    return hover_text_mapping

def edge_labels(edges: list["Edge"]) -> dict[tuple[str, str], str]:
    """Create a mapping from edges (source, target) to labels for visualization."""
    label_mapping = {}
    for edge in edges:
        source, target = edge.source, edge.target
        target_port, source_port = list(edge.ports_map.items())[0]
        label_mapping[(source, target)] = f"{source_port} -> {target_port}"  # No labels for now, but can be customized if needed
    return label_mapping

def edge_curved_mapping(edges: list["Edge"], node_configs: dict[str, tuple[type[Node], NodeConfig]]) -> dict[tuple[str, str], bool]:
    """Create a mapping from edges (source, target) to a boolean indicating whether the edge should be curved or not.
    
    If an edge leads to a metric node, it will be curved to visually separate it from the main execution flow.
    """
    curved_mapping = {}
    for edge in edges:
        source, target = edge.source, edge.target
        target_node_config = node_configs[target][1]
        if target_node_config.node_type == NodeType.METRIC:
            curved_mapping[(source, target)] = True
        else:
            curved_mapping[(source, target)] = False
    return curved_mapping


def render_pipeline_graph(pipeline: "Pipeline", title="Pipeline Graph", figsize=(12, 6)) -> None:
    """Render the pipeline graph using iplotx, with nodes arranged in columns based on their type and depth in the graph.
    Each node's metadata is displayed on hover.
    """
    graph = pipeline.graph.copy()  # Make a copy of the graph to avoid modifying the original
    graph = pipeline.graph
    node_configs = pipeline.config.nodes
    # Get the column assignments for each node
    columns = split_execution_graph_into_columns(graph=graph,
                                                 graph_wo_metrics=pipeline.graph_wo_metrics,
                                                 source_node=pipeline.get_source_node_name(),
                                                 sink_node=pipeline.get_sink_node_name())
    # Get the color mapping for nodes
    node_color_mapping = node_name_to_color_mapping(node_configs)
    # Get the marker style mapping for nodes
    node_marker_mapping = node_name_to_marker_style_mapping(node_configs)
    # Get the hover text mapping for nodes
    node_hover_text_mapping = node_name_to_hover_text_mapping(node_configs)
    # Get the edge color and linestyle mapping
    edge_color_mapping = edge_to_color_mapping(pipeline.edges, node_configs)
    edge_linestyle_map = edge_to_linestyle_mapping(pipeline.edges, node_configs)
    edge_curved_map = edge_curved_mapping(pipeline.edges, node_configs)
    # Get the hover text mapping for edges
    edge_hover_text_mapping = edge_to_hover_text_mapping(pipeline.edges)
    # Get the node labels
    node_labels = {node: str(node) for node in graph.nodes}
    edge_lab= edge_labels(pipeline.edges)  # No edge labels for now, but can be customized if needed
    # Create the iplotx figures
    pos = nx.multipartite_layout(graph, subset_key=columns)
    fig, ax = plt.subplots(figsize=figsize)
    artist = ipx.plot(
        graph,
        layout=pos,
        style={
            "vertex": {
                "facecolor": node_color_mapping,
                "marker": node_marker_mapping,
                "label": {"verticalalignment": "top", "horizontalalignment": "center", "fontsize": 8, "color": "black"},
            },
            "edge": {
                "color": edge_color_mapping,
                "linestyle": edge_linestyle_map,
                "paralleloffset": 0.1,
                "label": {
                    "rotate": False,
                    "fontsize": 5,
                    "color": "black",
                },
                "alpha": 0.8,

            },
        },
        node_labels=node_labels,
        edge_labels=edge_lab,
        ax=ax
    )
    ax.set_title(title)
    ax.set_axis_off()

    # Add hover tooltips for nodes
    nodes_in_order = list(graph.nodes)
    xs = [float(pos[n][0]) for n in nodes_in_order]
    ys = [float(pos[n][1]) for n in nodes_in_order]

    hover_points = ax.scatter(xs, ys, s=300, alpha=0.0)
    cursor = mplcursors.cursor(hover_points, hover=2) # Transient hover (disappears when cursor moves away from the points)
    @cursor.connect("add")
    def on_add(sel):
        idx = sel.index
        node = nodes_in_order[idx]
        sel.annotation.set_text(node_hover_text_mapping[node])
        sel.annotation.get_bbox_patch().set(alpha=0.95)

    plt.tight_layout()
    return fig, ax, artist


# TODO : add edge color depending on the execution mode of the connexion, and labels for the nodes without hover, create prettier hovertext

def _dict_to_hover_html(data: Any, indent: int = 0) -> str:
    """Format nested dict/list data as HTML for Plotly hover."""
    pad = "&nbsp;" * 4 * indent

    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            key = html.escape(str(k))
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{pad}<b>{key}</b>:")
                lines.append(_dict_to_hover_html(v, indent + 1))
            else:
                val = html.escape(str(v))
                lines.append(f"{pad}<b>{key}</b>: {val}")
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


def _edge_hover_html(source: str, target: str, ports_map: dict[str, str]) -> str:
    lines = [f"<b>{html.escape(source)} → {html.escape(target)}</b>", "<b>Ports mapping</b>"]
    for target_port, source_port in ports_map.items():
        lines.append(
            f"{html.escape(str(source_port))} → {html.escape(str(target_port))}"
        )
    return "<br>".join(lines)


import numpy as np


def _vertical_center(pos: dict[str, Any]) -> float:
    ys = [float(coords[1]) for coords in pos.values()]
    return (min(ys) + max(ys)) / 2.0


def _signed_curvature_from_source_y(
    source_y: float,
    y_center: float,
    min_curvature: float = 0.10,
    max_curvature: float = 0.45,
    power: float = 1.2,
) -> float:
    """Curvature sign and magnitude based on source node vertical position.

    - Above center  -> positive curvature
    - Below center  -> negative curvature
    - Farther from center -> larger absolute curvature
    """
    dist = abs(source_y - y_center)

    # Normalize distance to [0, 1], guarding against zero division
    max_dist = max(abs(y_center - source_y), 1e-12)
    # This local normalization is not enough by itself, so this helper is mainly
    # intended to be used through `_edge_curvature(...)` below.
    scaled = dist / max_dist
    scaled = min(max(scaled, 0.0), 1.0)

    mag = min_curvature + (max_curvature - min_curvature) * (scaled ** power)
    sign = 1.0 if source_y >= y_center else -1.0
    return sign * mag


def _edge_curvature(
    source: str,
    pos: dict[str, Any],
    y_center: float,
    min_curvature: float = 0.18,
    max_curvature: float = 0.45,
    power: float = 1.2,
) -> float:
    """Compute signed curvature for one edge from the source node position.

    The farther the source is from the vertical center, the stronger the curvature.
    """
    source_y = float(pos[source][1])

    ys = [float(coords[1]) for coords in pos.values()]
    max_dist = max(max(abs(y - y_center) for y in ys), 1e-12)

    norm_dist = abs(source_y - y_center) / max_dist
    norm_dist = min(max(norm_dist, 0.0), 1.0)

    mag = min_curvature + (max_curvature - min_curvature) * (norm_dist ** power)

    # Below center => negative curvature
    # Above center => positive curvature
    sign = 1.0 if source_y >= y_center else -1.0

    return sign * mag

def _quadratic_bezier_point_and_tangent(p0, p1, curvature=0.18, t=0.5):
    x0, y0 = p0
    x1, y1 = p1

    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0

    # perpendicular direction for control point
    nx_, ny_ = -dy, dx
    norm = (nx_**2 + ny_**2) ** 0.5 or 1.0
    nx_, ny_ = nx_ / norm, ny_ / norm

    dist = (dx**2 + dy**2) ** 0.5
    cx, cy = mx + curvature * dist * nx_, my + curvature * dist * ny_

    # point on quadratic bezier
    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1

    # derivative of quadratic bezier
    tx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
    ty = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)

    return (x, y), (tx, ty)


def _quadratic_bezier_points(p0, p1, curvature=0.18, steps=80):
    xs, ys = [], []
    for i in range(steps + 1):
        t = i / steps
        (x, y), _ = _quadratic_bezier_point_and_tangent(p0, p1, curvature=curvature, t=t)
        xs.append(x)
        ys.append(y)
    return xs, ys


def _quadratic_bezier_control_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1

    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy) or 1.0

    # unit normal
    nx_, ny_ = -dy / dist, dx / dist

    cx = mx + curvature * dist * nx_
    cy = my + curvature * dist * ny_
    return cx, cy


def _quadratic_bezier_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    c: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = c

    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
    return x, y


def _quadratic_bezier_derivative(
    p0: tuple[float, float],
    p1: tuple[float, float],
    c: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = c

    dx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
    dy = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)
    return dx, dy


def _quadratic_bezier_polyline(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
    steps: int = 200,
) -> tuple[list[float], list[float], tuple[float, float], np.ndarray]:
    c = _quadratic_bezier_control_point(p0, p1, curvature)
    ts = np.linspace(0.0, 1.0, steps + 1)

    xs = []
    ys = []
    for t in ts:
        x, y = _quadratic_bezier_point(p0, p1, c, float(t))
        xs.append(x)
        ys.append(y)

    pts = np.column_stack([xs, ys])
    return xs, ys, c, ts


def _bezier_t_for_arc_fraction(
    p0: tuple[float, float],
    p1: tuple[float, float],
    c: tuple[float, float],
    steps: int = 400,
    arc_fraction: float = 0.5,
) -> float:
    """Approximate the parameter t corresponding to a target fraction of arc length.
    """
    ts = np.linspace(0.0, 1.0, steps + 1)
    pts = np.array([_quadratic_bezier_point(p0, p1, c, float(t)) for t in ts])

    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]

    if total == 0:
        return 0.5

    target = arc_fraction * total
    idx = int(np.searchsorted(cum, target))
    idx = max(1, min(idx, len(ts) - 1))

    prev_len = cum[idx - 1]
    next_len = cum[idx]
    alpha = 0.0 if next_len == prev_len else (target - prev_len) / (next_len - prev_len)

    return float(ts[idx - 1] + alpha * (ts[idx] - ts[idx - 1]))


def _sample_quadratic_bezier_with_exact_label_pose(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
    steps: int = 200,
    label_arc_fraction: float = 0.5,
) -> tuple[list[float], list[float], float, float, float]:
    """Sample curve for plotting, but compute label point and angle from the exact
    quadratic derivative evaluated at the arc-length-based t.
    """
    xs, ys, c, _ = _quadratic_bezier_polyline(
        p0=p0,
        p1=p1,
        curvature=curvature,
        steps=steps,
    )

    t_label = _bezier_t_for_arc_fraction(
        p0=p0,
        p1=p1,
        c=c,
        steps=max(400, steps),
        arc_fraction=label_arc_fraction,
    )

    lx, ly = _quadratic_bezier_point(p0, p1, c, t_label)
    dx, dy = _quadratic_bezier_derivative(p0, p1, c, t_label)

    angle = math.degrees(math.atan2(dy, dx))

    # keep text upright
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return xs, ys, lx, ly, angle

def _arc_length_mid_t(xs, ys):
    pts = np.column_stack([xs, ys])
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    half = cum[-1] / 2.0
    idx = int(np.searchsorted(cum, half))

    if idx <= 0:
        return 0.0
    if idx >= len(cum):
        return 1.0

    prev_len = cum[idx - 1]
    next_len = cum[idx]
    alpha = 0.0 if next_len == prev_len else (half - prev_len) / (next_len - prev_len)

    # map segment location back to bezier t in [0,1]
    steps = len(xs) - 1
    t0 = (idx - 1) / steps
    t1 = idx / steps
    return t0 + alpha * (t1 - t0)


def _curve_label_pose_from_points(
    xs: list[float],
    ys: list[float],
    curvature: float,
    normal_offset: float = 0.035,
) -> tuple[float, float, float]:
    """Compute label position/angle directly from the already-sampled curve points.

    This guarantees the label follows the exact rendered curve.
    """
    pts = np.column_stack([xs, ys])

    if len(pts) < 2:
        return xs[0], ys[0], 0.0

    # Arc-length midpoint
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    half = cum[-1] / 2.0
    mid_idx = int(np.searchsorted(cum, half))
    mid_idx = max(1, min(mid_idx, len(xs) - 2))

    # Interpolate midpoint on the segment
    prev_len = cum[mid_idx - 1]
    next_len = cum[mid_idx]
    alpha = 0.0 if next_len == prev_len else (half - prev_len) / (next_len - prev_len)

    x = xs[mid_idx - 1] + alpha * (xs[mid_idx] - xs[mid_idx - 1])
    y = ys[mid_idx - 1] + alpha * (ys[mid_idx] - ys[mid_idx - 1])

    # Local tangent from neighboring sampled points
    x0, y0 = xs[mid_idx - 1], ys[mid_idx - 1]
    x1, y1 = xs[mid_idx + 1], ys[mid_idx + 1]
    tx, ty = x1 - x0, y1 - y0

    norm = math.hypot(tx, ty) or 1.0
    tx, ty = tx / norm, ty / norm

    # Normal
    nx_, ny_ = -ty, tx

    # Put the label on the outside of the bend in a sign-aware way
    side = -1.0 if curvature > 0 else 1.0
    x += side * normal_offset * nx_
    y += side * normal_offset * ny_

    angle = math.degrees(math.atan2(ty, tx))

    # Keep text upright
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return x, y, angle


def _edge_midpoint_for_label(p0, p1, curved: bool = False, curvature: float = 0.18):
    if not curved:
        return (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    xs, ys = _quadratic_bezier_points(p0, p1, curvature=curvature, steps=24)
    mid = len(xs) // 2
    return xs[mid], ys[mid]




def _add_edge_shapes_with_labels(
    fig: go.Figure,
    pipeline: "Pipeline",
    pos: dict[str, Any],
    edge_color_mapping: dict[tuple[str, str], str],
    edge_linestyle_map: dict[tuple[str, str], str],
    edge_label_map: dict[tuple[str, str], str],
    edge_curved_map: dict[tuple[str, str], bool],
) -> None:
    dash_map = {
        "-": "solid",
        "--": "dash",
        "-.": "dashdot",
        ":": "dot",
    }

    for edge in pipeline.edges:
        source = edge.source
        target = edge.target

        x0, y0 = float(pos[source][0]), float(pos[source][1])
        x1, y1 = float(pos[target][0]), float(pos[target][1])

        curved = edge_curved_map.get((source, target), False)
        label_text = edge_label_map.get((source, target), "")

        # Only straight edges here
        if curved:
            continue

        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(
                color=edge_color_mapping.get((source, target), "gray"),
                width=2,
                dash=dash_map.get(edge_linestyle_map.get((source, target), "-"), "solid"),
            ),
            label=dict(
                text=label_text,
                textposition="middle",
                padding=6,
                font=dict(size=9, color="black"),
            ),
            layer="below",
        )


def _sample_quadratic_bezier_with_label_pose(
    p0: tuple[float, float],
    p1: tuple[float, float],
    curvature: float,
    steps: int = 160,
    label_arc_fraction: float = 0.5,
) -> tuple[list[float], list[float], float, float, float]:
    """Build the rendered curve and compute the label position/angle from the same sampled points.

    Returns
    -------
    xs, ys, lx, ly, angle_deg

    """
    x0, y0 = p0
    x1, y1 = p1

    # Control point from signed perpendicular offset
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy) or 1.0

    nx_, ny_ = -dy / dist, dx / dist
    cx = mx + curvature * dist * nx_
    cy = my + curvature * dist * ny_

    # Sample the exact polyline that will be rendered
    ts = np.linspace(0.0, 1.0, steps + 1)
    xs = ((1 - ts) ** 2) * x0 + 2 * (1 - ts) * ts * cx + (ts ** 2) * x1
    ys = ((1 - ts) ** 2) * y0 + 2 * (1 - ts) * ts * cy + (ts ** 2) * y1

    xs = xs.tolist()
    ys = ys.tolist()

    # Arc-length parameterization on the sampled polyline
    pts = np.column_stack([xs, ys])
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] if len(cum) else 0.0

    target = label_arc_fraction * total
    idx = int(np.searchsorted(cum, target))
    idx = max(1, min(idx, len(xs) - 2))

    # Interpolate label point on the located segment
    prev_len = cum[idx - 1]
    next_len = cum[idx]
    alpha = 0.0 if next_len == prev_len else (target - prev_len) / (next_len - prev_len)

    lx = xs[idx - 1] + alpha * (xs[idx] - xs[idx - 1])
    ly = ys[idx - 1] + alpha * (ys[idx] - ys[idx - 1])

    # Tangent from the same sampled polyline
    tx = xs[idx + 1] - xs[idx - 1]
    ty = ys[idx + 1] - ys[idx - 1]
    angle = math.degrees(math.atan2(ty, tx))

    # Keep text upright
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return xs, ys, lx, ly, angle

def _add_legend(fig, graph, pipeline, node_configs,
                node_color_mapping,
                node_marker_mapping,
                edge_color_mapping):

    # ---- Node legend (only present types)
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

    seen_node_types = {}

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
                    "symbol": marker_symbol_map.get(node_marker_mapping[node], "circle"),
                    "line": {"color": "black", "width": 1},
                },
                name=str(node_type.name),  # cleaner label
                legendgroup="nodes",
                showlegend=True,
            )
        )

    # ---- Edge legend (only present colors)
    color_to_label = {
        "black": "All modes",
        "blue": "Multiple modes",
        "green": "Train",
        "orange": "Inference",
        "purple": "Evaluation",
        "gray": "Other",
    }

    seen_colors = set()

    for edge, color in edge_color_mapping.items():
        if color in seen_colors:
            continue
        seen_colors.add(color)

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color, width=3),
                name=color_to_label.get(color, color),
                legendgroup="edges",
                showlegend=True,
            )
        )

def render_pipeline_graph_plotly(
    pipeline: "Pipeline",
    title: str = "Pipeline Graph",
    version: Version | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> go.Figure:
    if version is None:
        version = pipeline.config.version
    graph = pipeline.graph
    node_configs = pipeline.config.nodes

    columns = split_execution_graph_into_columns(
        graph=graph,
        graph_wo_metrics=pipeline.graph_wo_metrics,
        source_node=pipeline.get_source_node_name(),
        sink_node=pipeline.get_sink_node_name(),
    )

    pos = nx.multipartite_layout(graph, subset_key=columns)
    y_center = _vertical_center(pos)

    node_color_mapping = node_name_to_color_mapping(node_configs)
    node_marker_mapping = node_name_to_marker_style_mapping(node_configs)
    node_data_mapping = node_name_to_node_data_mapping(node_configs)

    edge_color_mapping = edge_to_color_mapping(pipeline.edges, node_configs)
    edge_linestyle_map = edge_to_linestyle_mapping(pipeline.edges, node_configs)
    edge_curved_map = edge_curved_mapping(pipeline.edges, node_configs)
    edge_label_map = edge_labels(pipeline.edges)

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

    fig = go.Figure()

    # Hoverable edge traces only where needed:
    # - all curved edges
    # - optionally all edges if you want hover everywhere
    dash_map = {
        "-": "solid",
        "--": "dash",
        "-.": "dashdot",
        ":": "dot",
    }

    for edge in pipeline.edges:
        source = edge.source
        target = edge.target
        ports_map = edge.ports_map

        x0, y0 = float(pos[source][0]), float(pos[source][1])
        x1, y1 = float(pos[target][0]), float(pos[target][1])

        curved = edge_curved_map.get((source, target), False)

        if curved:
            curvature = _edge_curvature(
                source=source,
                pos=pos,
                y_center=y_center,
                min_curvature=0.10,
                max_curvature=0.45,
                power=1.2,
            )

            xs, ys, lx, ly, angle = _sample_quadratic_bezier_with_exact_label_pose(
                (x0, y0),
                (x1, y1),
                curvature=curvature,
                steps=200,
                label_arc_fraction=0.5,
            )

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={
                        "color": edge_color_mapping.get((source, target), "gray"),
                        "width": 2,
                        "dash": dash_map.get(edge_linestyle_map.get((source, target), "-"), "solid"),
                    },
                    hovertemplate=_edge_hover_html(source, target, edge.ports_map) + "<extra></extra>",
                    showlegend=False,
                )
            )

            fig.add_annotation(
                x=lx,
                y=ly,
                text=edge_label_map.get((source, target), ""),
                showarrow=False,
                textangle=angle,
                font={"size": 9, "color": "black"},
                xanchor="center",
                yanchor="middle",
                bgcolor="rgba(255,255,255,0.0)",
                borderpad=0,
            )
        else:
            # invisible hover trace for straight edge
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={
                        "color": "rgba(0,0,0,0)",
                        "width": 10,
                    },
                    hovertemplate=_edge_hover_html(source, target, ports_map) + "<extra></extra>",
                    showlegend=False,
                )
            )

    # Visible edge lines + tangent labels
    _add_edge_shapes_with_labels(
        fig=fig,
        pipeline=pipeline,
        pos=pos,
        edge_color_mapping=edge_color_mapping,
        edge_linestyle_map=edge_linestyle_map,
        edge_label_map=edge_label_map,
        edge_curved_map=edge_curved_map,
    )

    # Nodes
    nodes_in_order = list(graph.nodes)
    node_x = [float(pos[n][0]) for n in nodes_in_order]
    node_y = [float(pos[n][1]) for n in nodes_in_order]
    node_hover = [_node_hover_html(n, node_data_mapping.get(n, {})) for n in nodes_in_order]
    node_symbols = [marker_symbol_map.get(node_marker_mapping.get(n, "o"), "circle") for n in nodes_in_order]
    node_colors = [node_color_mapping.get(n, "white") for n in nodes_in_order]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(n) for n in nodes_in_order],
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
        )
    )

    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    fig.update_layout(
        title={
            "text": f"{title}<br><span style='font-size:12px;color:gray'>{version!s}</span>",
            "x": 0.5,
            "xanchor": "center"
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
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
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
        }
    )
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=0),
        name="— Nodes —",
        showlegend=True
    ))

    _add_legend(
        fig=fig,
        graph=graph,
        pipeline=pipeline,
        node_configs=node_configs,
        node_color_mapping=node_color_mapping,
        node_marker_mapping=node_marker_mapping,
        edge_color_mapping=edge_color_mapping,
    )

    return fig
