
from typing import Any

import iplotx as ipx
import matplotlib.pyplot as plt
import mplcursors
import networkx as nx
from pydantic import BaseModel

from tsut.core.nodes.node import Node


def plot_graph_with_metadata(
    G: nx.Graph,
    node_objects: dict[str, Node],
    title: str = "Pipeline Graph",
    layout: str = "spring",
    figsize: tuple[int, int] = (10, 8),
):
    """Render a NetworkX graph with iplotx and show each node's metadata on hover.

    Assumptions
    -----------
    - G is a networkx graph
    - node_objects maps node_id -> object
    - each object has a `.metadata` attribute
    - `.metadata` is a Pydantic model (v2: model_dump, v1: dict)

    Returns
    -------
    (fig, ax, artist)
        Matplotlib figure, axes, and the iplotx artist.

    """

    def _metadata_to_dict(metadata: BaseModel | None) -> dict[str, Any]:
        if metadata is None:
            return {}
        if hasattr(metadata, "model_dump"):   # Pydantic v2
            return metadata.model_dump()
        if hasattr(metadata, "dict"):         # Pydantic v1
            return metadata.dict()
        if isinstance(metadata, dict):
            return metadata
        return {"value": str(metadata)}

    def _format_hover_text(node: Any) -> str:
        obj = node_objects.get(node)
        if obj is None:
            return f"node: {node}\n(no associated object)"

        metadata = getattr(obj, "metadata", None)
        meta_dict = _metadata_to_dict(metadata)

        if not meta_dict:
            return f"node: {node}\n(no metadata)"

        lines = [f"node: {node}"]
        for k, v in meta_dict.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def _compute_layout(graph: nx.Graph, layout_name: str):
        if layout_name == "spring":
            return nx.spring_layout(graph, seed=42)
        if layout_name == "kamada_kawai":
            return nx.kamada_kawai_layout(graph)
        if layout_name == "circular":
            return nx.circular_layout(graph)
        if layout_name == "shell":
            return nx.shell_layout(graph)
        if layout_name == "spectral":
            return nx.spectral_layout(graph)
        if layout_name == "random":
            return nx.random_layout(graph, seed=42)
        raise ValueError(
            f"Unsupported layout '{layout_name}'. "
            "Use one of: spring, kamada_kawai, circular, shell, spectral, random."
        )

    # Make sure we use a graph copy so we do not mutate the original
    G_plot = G.copy()

    # Precompute text for each node
    hover_text_by_node = {node: _format_hover_text(node) for node in G_plot.nodes}

    # Optional: add compact visible labels
    for node in G_plot.nodes:
        G_plot.nodes[node]["label"] = str(node)

    # IMPORTANT: pass a real layout object to iplotx, not a string
    pos = _compute_layout(G_plot, layout)

    fig, ax = plt.subplots(figsize=figsize)

    # Conservative styling that follows documented ipx.plot(..., layout=pos, style=...)
    artist = ipx.plot(
        G_plot,
        layout=pos,
        vertex_labels=True,
        style={
            "vertex": {
                "size": 18,
                "facecolor": "lightsteelblue",
                "edgecolor": "black",
                "linewidth": 1.0,
                "label": {  # 👈 THIS is the correct way
                    "color": "black",
                    "fontsize": 10,
                },
            },
            "edge": {
                "alpha": 0.7,
                "linewidth": 1.2,
            },
        },
        ax=ax,
    )

    ax.set_title(title)
    ax.set_axis_off()

    # Add invisible scatter points exactly at node positions so mplcursors
    # can provide reliable hover tooltips regardless of iplotx internals.
    nodes_in_order = list(G_plot.nodes)
    xs = [float(pos[n][0]) for n in nodes_in_order]
    ys = [float(pos[n][1]) for n in nodes_in_order]

    hover_points = ax.scatter(xs, ys, s=300, alpha=0.0)

    cursor = mplcursors.cursor(hover_points, hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        idx = sel.index
        node = nodes_in_order[idx]
        sel.annotation.set_text(hover_text_by_node[node])
        sel.annotation.get_bbox_patch().set(alpha=0.95)

    plt.tight_layout()
    return fig, ax, artist
