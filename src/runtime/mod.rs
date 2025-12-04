//! Runtime module: executes the graph.

use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::ops::registry::OpRegistry;
use crate::onnx::onnx_proto::{type_proto, tensor_shape_proto};
use std::collections::HashMap;

pub struct InferenceSession {
    pub graph: Graph,
    pub registry: OpRegistry,
}

impl InferenceSession {
    pub fn new(graph: Graph) -> anyhow::Result<Self> {
        if graph.inputs.len() != 1 {
            return Err(anyhow::anyhow!("Expect model to have exactly 1 input, got {}: {:?}", graph.inputs.len(), graph.inputs));
        }
        Ok(Self {
            graph,
            registry: OpRegistry::new(),
        })
    }

    fn validate_inputs(&self, inputs: &[Tensor]) -> anyhow::Result<HashMap<String, Tensor>> {
        let mut values = HashMap::new();
        for (idx, param) in self.graph.proto.input.iter().enumerate() {
            // Skip inputs that are initializers (weights)
            if self.graph.initializers.contains_key(&param.name) {
                continue;
            }

            let input_tensor = inputs.get(idx).ok_or_else(|| {
                anyhow::anyhow!("Missing input tensor at index {}", idx)
            })?;

            // Parse expected shape from TypeProto
            if let Some(type_proto) = &param.r#type {
                if let Some(type_proto::Value::TensorType(tensor_type)) = &type_proto.value {
                    if let Some(shape_proto) = &tensor_type.shape {
                        let expected_shape: Vec<Option<usize>> = shape_proto.dim.iter()
                            .map(|dim| {
                                match &dim.value {
                                    Some(tensor_shape_proto::dimension::Value::DimValue(v)) => {
                                        Some(*v as usize)
                                    }
                                    Some(tensor_shape_proto::dimension::Value::DimParam(_)) => {
                                        // Symbolic dimension (e.g., "batch"), accept any value
                                        None
                                    }
                                    None => None,
                                }
                            })
                            .collect();

                        let actual_shape = input_tensor.shape();

                        // Check rank
                        if expected_shape.len() != actual_shape.len() {
                            return Err(anyhow::anyhow!(
                                "Input '{}': rank mismatch, expected {} dimensions, got {}",
                                param.name, expected_shape.len(), actual_shape.len()
                            ));
                        }

                        // Check each dimension
                        for (i, (expected, actual)) in expected_shape.iter().zip(actual_shape.iter()).enumerate() {
                            if let Some(exp) = expected {
                                if *exp != *actual {
                                    return Err(anyhow::anyhow!(
                                        "Input '{}': dimension {} mismatch, expected {}, got {}",
                                        param.name, i, exp, actual
                                    ));
                                }
                            }
                        }
                        values.insert(param.name.clone(), input_tensor.clone());
                    }
                }
            }
        }
        Ok(values)
    }

    pub fn run(&self, input: &[Tensor]) -> anyhow::Result<Vec<Tensor>> {
        // 1. Context: Map names to Tensors
        // This HashMap holds all live values (inputs + intermediate activations)
        let mut values = self.validate_inputs(&input)?;

        // 2. Execute nodes
        // NOTE: We assume by the spec that the graph is a valid DAG, already topologically sorted.
        // TODO: We might want to resort the graph and raise an error if it's not a valid DAG.
        for node in &self.graph.nodes {
            // println!("State: {:?}", values);
            // Gather inputs for this node
            let mut node_inputs = Vec::new();
            for input_name in &node.input {
                // First check computed values (activations), then constant weights (initializers)
                if let Some(t) = values.get(input_name) {
                    node_inputs.push(t);
                } else if let Some(t) = self.graph.initializers.get(input_name) {
                    node_inputs.push(t);
                } else {
                    return Err(anyhow::anyhow!("Missing input '{}' for node '{}'", input_name, node.name));
                }
            }

            println!("Running operator: {:#?}", node);
            // Dispatch to operator using the registry
            let op = self.registry.get(&node.op_type)
                .ok_or_else(|| anyhow::anyhow!("Unsupported operator: {}", node.op_type))?;
            
            let output = op.run(&node_inputs, node)?;

            // Store output (assuming single output for these simple ops)
            if let Some(output_name) = node.output.first() {
                values.insert(output_name.clone(), output);
            }
        }

        // 3. Return requested outputs
        let mut results = Vec::new();
        for output_name in &self.graph.outputs {
            let t = values.get(output_name)
                .ok_or_else(|| anyhow::anyhow!("Output '{}' not produced", output_name))?;
            // We clone the result to return ownership to the caller
            results.push(t.clone());
        }

        Ok(results)
    }
}
