Overview
========

TimeSeriesUnifiedToolbox (TSUT) organizes workflows around:

- **Core abstractions**: strongly-typed data contracts, node configuration, and pipeline graph semantics.
- **Composable nodes**: data sources, transforms, models, and metrics.
- **Pipeline execution**: graph compilation and execution through pipeline runners.

Core package layout
-------------------

- ``tsut.core.common``: data structures, common helpers, and registries.
- ``tsut.core.nodes``: base node definitions and typed port models.
- ``tsut.core.pipeline``: pipeline data model, compilation, rendering, and runners.
- ``tsut.components``: concrete, discoverable node implementations.
