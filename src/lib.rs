//! NeurOxyde: A minimal ONNX inference engine written in Rust.
//!
//! This library provides the core components for loading and executing ONNX models.

pub mod loader;
pub mod tensor;
pub mod ops;
pub mod graph;
pub mod runtime;
pub mod utils;

// Re-export generated ONNX bindings
// The generated file name depends on the package name in the proto file.
// Assuming "package onnx;" in onnx.proto3, prost usually generates onnx.rs.
// However, since we are outputting to src/onnx_generated, we need to include it.
// We'll wrap it in a module.

pub mod onnx {
    // This includes the generated code. 
    // We use include! macro to include the file from src/onnx_generated/
    
    #[allow(clippy::all)]
    pub mod onnx_proto {
        include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/onnx_generated/onnx.rs"));
    }
}

