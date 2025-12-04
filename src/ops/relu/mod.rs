use crate::ops::operator::Operator;
use crate::tensor::Tensor;
use crate::onnx::onnx_proto::NodeProto;

pub struct Relu;

impl Operator for Relu {
    fn run(&self, inputs: &[&Tensor], _node: &NodeProto) -> anyhow::Result<Tensor> {
        let input = inputs[0];
        let new_data: Vec<f32> = input.data()
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();
        
        Ok(Tensor::new(new_data, input.shape().to_vec()))
    }
}
