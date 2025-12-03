//! Loader module: handles loading ONNX models from files.

// use onnx::onnx::ModelProto;
use std::path::Path;
use thiserror::Error;
use std::fs::File;
use std::io::Read;
use prost::Message;
use crate::onnx::onnx_proto::ModelProto;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("Protobuf decode error")]
    Decode(#[from] prost::DecodeError),
}

pub struct ModelLoader {
    // Placeholder for loader state
}

impl ModelLoader {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<ModelProto> {
        // Read ONNX file into bytes
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Parse the ONNX protobuf
        let model = ModelProto::decode(&*buffer)?;

        // Print some basic metadata
        println!("Model IR version: {:?}", model.ir_version);
        println!("Producer: {:?}", model.producer_name);
        println!("Domain: {:?}", model.domain);
        println!("Model version: {:?}", model.model_version);
        println!("Model: {:#?}", model);
        // model.

        // Access the graph
        if let Some(graph) = &model.graph.as_ref() {
            println!("Nodes in graph: {}", graph.node.len());
            for (i, node) in graph.node.iter().enumerate() {
                println!("Node {} → op_type: {}", i, node.op_type);
            }
        }

        Ok(model)
    }
}

