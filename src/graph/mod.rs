//! Graph module: represents the computational graph.

use crate::tensor;
#[allow(unused_imports)]
use crate::ops::Operator;
use std::collections::HashMap;
use crate::onnx::onnx_proto::NodeProto;
use std::fmt;
use crate::loader::ModelLoader;

pub struct Graph {
    // Placeholder for graph structure
    pub nodes: Vec<NodeProto>,
    pub initializers: HashMap<String, tensor::Tensor>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl Graph {
    pub fn from_model(model: &ModelLoader) -> anyhow::Result<Self> {
        let g = model.model.graph.as_ref().ok_or(anyhow::anyhow!("Model has no graph"))?;

        // extract initializers into HashMap<String, Tensor>
        let mut inits = HashMap::new();
        for init in &g.initializer {
            let name = init.name.clone();
            let tensor = tensor::Tensor::from_proto(init);
            inits.insert(name, tensor);
        }

        let inputs = g.input.iter().map(|vi| vi.name.clone()).collect();
        let outputs = g.output.iter().map(|vi| vi.name.clone()).collect();

        Ok(Self {
            nodes: g.node.clone(),
            initializers: inits,
            inputs,
            outputs,
        })
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph:")?;
        writeln!(f, "  Inputs: {:?}", self.inputs)?;
        writeln!(f, "  Outputs: {:?}", self.outputs)?;

        writeln!(f, "  Initializers:")?;
        let mut init_names: Vec<_> = self.initializers.keys().collect();
        init_names.sort();
        for name in init_names {
            if let Some(tensor) = self.initializers.get(name) {
                writeln!(f, "    - {}: {:?}", name, tensor.shape())?;
            }
        }

        writeln!(f, "  Nodes:")?;
        for (i, node) in self.nodes.iter().enumerate() {
            let name = if node.name.is_empty() {
                format!("node_{}", i)
            } else {
                node.name.clone()
            };
            writeln!(
                f,
                "    - [{}] {}: ({}) -> ({})",
                node.op_type,
                name,
                node.input.join(", "),
                node.output.join(", ")
            )?;
        }
        Ok(())
    }
}
