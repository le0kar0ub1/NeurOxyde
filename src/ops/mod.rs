//! Ops module: implements mathematical operators and activation functions.

use crate::tensor::Tensor;

pub trait Operator {
    fn run(&self, inputs: &[&Tensor]) -> Tensor;
}

pub struct Add;

impl Operator for Add {
    fn run(&self, _inputs: &[&Tensor]) -> Tensor {
        todo!("Add operator implementation not written yet")
    }
}

pub struct Relu;

impl Operator for Relu {
    fn run(&self, _inputs: &[&Tensor]) -> Tensor {
        todo!("Relu operator implementation not written yet")
    }
}

