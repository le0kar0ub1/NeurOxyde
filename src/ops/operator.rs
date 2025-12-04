use crate::tensor::Tensor;
use crate::onnx::onnx_proto::NodeProto;

pub trait Operator {
    fn run(&self, inputs: &[&Tensor], node: &NodeProto) -> anyhow::Result<Tensor>;
}