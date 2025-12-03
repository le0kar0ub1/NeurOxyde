# NeurOxyde

NeurOxyde aims to be a high performance minimal inference engine written in pure Rust. It is designed to provide a lightweight runtime for executing Open Neural Network Exchange (ONNX) models with a focus on type safety and efficiency.

## Overview

NeurOxyde aims to bridge the gap between complex deep learning frameworks and embedded or edge deployment scenarios by offering a streamlined execution environment. It leverages Rust's memory safety guarantees and zero-cost abstractions to deliver reliable inference.

## Key Features

- **ONNX Support**: Native parsing and loading of standard ONNX model files.
- **Tensor Operations**: Efficient n-dimensional array manipulations.
- **Compute Graph**: Directed acyclic graph (DAG) representation for model execution.
- **Modular Design**: Clear separation between model loading, graph optimization, and runtime execution.

## Architecture

The project is organized into the following core modules:

- **Loader**: Handles the deserialization of `.onnx` protobuf files.
- **Tensor**: Provides the fundamental data structures for numerical computation.
- **Ops**: Implements mathematical operators (kernels) for inference.
- **Graph**: Manages the computational graph topology and node connectivity.
- **Runtime**: Orchestrates the execution flow and resource management.

## Getting Started

### Prerequisites

- Rust (latest stable toolchain)
- Cargo

### Building

NeurOxyde uses `prost` for Protocol Buffers code generation. The build process automatically handles the generation of Rust bindings from the ONNX schema.

```bash
cargo build --release
```

### Running Examples

To run the provided inference examples:

```bash
cargo run --example mnist_infer
```

## Development

The ONNX protocol buffer definitions are located in `onnx_proto/`. During the build process, `build.rs` compiles these definitions into Rust code located in `src/onnx_generated/`.

To regenerate bindings after updating the `.proto` file:

```bash
cargo clean
cargo build
```
