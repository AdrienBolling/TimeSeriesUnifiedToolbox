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

### 3. Node Execution Modes

The runner supports three execution modes:
- **`TRAIN`**: Fits/trains nodes and produces outputs for downstream nodes
- **`PREDICT`**: Only transforms data through nodes (no fitting)
- **`EVALUATE`**: Evaluates performance (placeholder for future implementation)

### 4. Critical Design Principle: Always Produce Outputs

**Key Insight**: In all execution modes, nodes must produce outputs because downstream nodes need these outputs as inputs, even during training.

This means:
- **TRAIN mode**: Calls `node_fit_transform()` - fits the node AND produces outputs
- **PREDICT mode**: Calls `node_transform()` - only transforms to produce outputs
- **EVALUATE mode**: Calls `node_transform()` - same as predict for now

### 5. Data Flow

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

During execution, the runner:
1. Retrieves the topologically sorted node order
2. Iterates through nodes in this order
3. For each node:
   - Collects input data from upstream nodes' outputs
   - Maps data according to edge port mappings
   - Executes the node's operation based on mode:
     - **TRAIN**: `node_fit_transform()` - fits and produces outputs
     - **PREDICT**: `node_transform()` - only produces outputs
     - **EVALUATE**: `node_transform()` - same as predict (placeholder)
   - Stores output data for downstream nodes

### 3. Data Management

The runner maintains:
- **Intermediate results**: Stores outputs from each node for downstream consumption
- **Context data**: Additional metadata that flows through the pipeline

### 4. Sink Nodes and Final Outputs

The runner provides a method to retrieve outputs from:
- **Sink nodes**: Nodes explicitly marked with `node_type = SINK`
- **Leaf nodes**: Nodes with no successors (end of the pipeline)

## Naive Implementation Strategy

A naive implementation focuses on correctness over optimization:

1. **Sequential Execution**: Execute nodes one at a time in topological order
2. **In-Memory Storage**: Store all intermediate results in memory
3. **Simple Data Passing**: Directly pass data between nodes without optimization
4. **No Parallelization**: Don't attempt concurrent execution

### Algorithm Pseudocode

```
function run_pipeline(pipeline, mode='train'):
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
        
        # Execute node based on mode - ALWAYS produces outputs
        if mode == 'train':
            # Fit and transform to produce outputs for downstream nodes
            outputs = node.node_fit_transform(inputs)
        elif mode == 'predict':
            # Only transform to produce outputs
            outputs = node.node_transform(inputs)
        else:  # evaluate
            # Same as predict for now (placeholder)
            outputs = node.node_transform(inputs)
        
        # Store outputs for downstream nodes
        node_outputs[node_name] = outputs
    
    return node_outputs

function get_sink_outputs(node_outputs):
    # Return outputs from sink or leaf nodes only
    sink_outputs = {}
    for node_name, node in pipeline.nodes:
        if node.type == 'sink' or has_no_successors(node_name):
            sink_outputs[node_name] = node_outputs[node_name]
    return sink_outputs
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
