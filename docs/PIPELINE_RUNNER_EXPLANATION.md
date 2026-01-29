# PipelineRunner Explanation

## Overview

The `PipelineRunner` is a component responsible for orchestrating the execution of a pipeline represented as a Directed Acyclic Graph (DAG). It ensures that nodes (processing units) are executed in the correct order, respecting their dependencies.

## Core Concepts

### 1. Pipeline as a DAG

A pipeline is modeled as a DAG where:
- **Nodes** represent processing units (data sources, transformations, models, etc.)
- **Edges** represent data flow and dependencies between nodes
- The acyclic property ensures no circular dependencies exist

### 2. Execution Order

The PipelineRunner must determine a valid execution order using **topological sorting**. This algorithm ensures:
- A node is executed only after all its dependencies have completed
- The execution respects the data flow direction
- Multiple valid orderings may exist, any one is acceptable

### 3. Node Execution Phases

Each node supports two primary operations:
- **`node_fit()`**: Fits/trains the node with input data (used during training)
- **`node_transform()`**: Transforms data through the node (used during inference)
- **`node_fit_transform()`**: Combines both operations

### 4. Data Flow

Data flows through the pipeline via ports:
- **Input ports**: Where a node receives data from upstream nodes
- **Output ports**: Where a node produces data for downstream nodes
- **Port mapping**: Edges specify which output port connects to which input port

## PipelineRunner Responsibilities

### 1. Compilation

Before execution, the runner must compile the pipeline:
- Validate the DAG structure
- Initialize all node objects
- Compute the execution order using topological sort

### 2. Execution Orchestration

During execution (e.g., training), the runner:
1. Retrieves the topologically sorted node order
2. Iterates through nodes in this order
3. For each node:
   - Collects input data from upstream nodes' outputs
   - Maps data according to edge port mappings
   - Executes the node's operation (fit, transform, or fit_transform)
   - Stores output data for downstream nodes

### 3. Data Management

The runner maintains:
- **Input data cache**: Stores initial data sources
- **Intermediate results**: Stores outputs from each node for downstream consumption
- **Context data**: Additional metadata that flows through the pipeline

## Naive Implementation Strategy

A naive implementation focuses on correctness over optimization:

1. **Sequential Execution**: Execute nodes one at a time in topological order
2. **In-Memory Storage**: Store all intermediate results in memory
3. **Simple Data Passing**: Directly pass data between nodes without optimization
4. **No Parallelization**: Don't attempt concurrent execution

### Algorithm Pseudocode

```
function run_pipeline(pipeline, mode='fit_transform'):
    # Compile if not already compiled
    if not pipeline.compiled():
        compile(pipeline)
    
    # Get execution order
    execution_order = topological_sort(pipeline.graph)
    
    # Initialize data storage
    node_outputs = {}
    
    # Execute each node in order
    for node_name in execution_order:
        node = pipeline.node_objects[node_name]
        
        # Gather inputs from upstream nodes
        inputs = {}
        for predecessor in get_predecessors(node_name):
            edge_data = get_edge_data(predecessor, node_name)
            for source_port, target_port in edge_data.ports_map:
                inputs[target_port] = node_outputs[predecessor][source_port]
        
        # Execute node
        if mode == 'fit':
            node.node_fit(inputs)
        elif mode == 'transform':
            outputs = node.node_transform(inputs)
        else:  # fit_transform
            outputs = node.node_fit_transform(inputs)
        
        # Store outputs
        if mode != 'fit':
            node_outputs[node_name] = outputs
    
    return node_outputs
```

## Future Enhancements

The naive implementation can be enhanced with:
- **Parallel Execution**: Execute independent nodes concurrently
- **Lazy Evaluation**: Only compute necessary paths
- **Caching**: Avoid recomputation of unchanged nodes
- **Distributed Execution**: Leverage Ray for distributed processing
- **Checkpointing**: Save intermediate states for fault tolerance
- **Resource Management**: Optimize memory by releasing unused intermediate results

## Integration with Existing Code

The PipelineRunner integrates with:
- `Pipeline`: Provides the DAG structure and node objects
- `Node`: Executes the actual processing logic
- `Edge`: Defines data flow connections
- `Port`: Specifies input/output interfaces

The runner uses NetworkX's `topological_sort` function for ordering, which is already a dependency in the pipeline module.
