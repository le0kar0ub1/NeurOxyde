//! Runtime module: executes the graph.

use crate::graph::Graph;
use crate::tensor::Tensor;

pub struct InferenceSession {
    pub graph: Graph,
}

impl InferenceSession {
    pub fn new(_graph: Graph) -> Self {
        todo!("InferenceSession implementation not written yet")
    }

    pub fn run(&self, _inputs: Vec<Tensor>) -> Vec<Tensor> {
        todo!("InferenceSession::run implementation not written yet")
    }
}

