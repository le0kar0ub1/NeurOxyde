//! Loader module: handles loading ONNX models from files.

// use onnx::onnx::ModelProto;
use std::path::Path;
use thiserror::Error;
use std::fs::File;
use std::io::Read;
use prost::Message;
use crate::onnx::onnx_proto::ModelProto;
use std::fmt;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("Protobuf decode error")]
    Decode(#[from] prost::DecodeError),
}

pub struct ModelLoader {
    pub model: ModelProto,
}

impl ModelLoader {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        // Read ONNX file into bytes
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Parse the ONNX protobuf
        let model = ModelProto::decode(&*buffer)?;

        Ok(Self { model })
    }
}

impl fmt::Display for ModelLoader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model IR version: {:?}", self.model.ir_version)?;
        writeln!(f, "Producer: {:?}", self.model.producer_name)?;
        writeln!(f, "Domain: {:?}", self.model.domain)?;
        writeln!(f, "Model version: {:?}", self.model.model_version)?;
        writeln!(f, "Model: {:#?}", self.model)?;
        // model.

        // Access the graph
        if let Some(graph) = &self.model.graph.as_ref() {
            writeln!(f, "Nodes in graph: {}", graph.node.len())?;
            for (i, node) in graph.node.iter().enumerate() {
                writeln!(f, "Node {} → op_type: {}", i, node.op_type)?;
            }
        }
        Ok(())
    }
}