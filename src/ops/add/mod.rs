use crate::ops::operator::Operator;
use crate::tensor::Tensor;
use crate::onnx::onnx_proto::NodeProto;

pub struct Add;

impl Operator for Add {
  fn run(&self, inputs: &[&Tensor], _node: &NodeProto) -> anyhow::Result<Tensor> {
      // Basic Add implementation: assumes broadcasting or same shape
      // For MVP, we assume same shape
      let a = inputs[0];
      let b = inputs[1];
      
      // Very basic broadcasting check (incomplete)
      if a.shape() != b.shape() {
          // Check for scalar broadcasting (common case)
           if b.shape().len() == 0 || (b.shape().len() == 1 && b.shape()[0] == 1) {
               let scalar = b.data()[0];
               let new_data: Vec<f32> = a.data().iter().map(|&x| x + scalar).collect();
               return Ok(Tensor::new(new_data, a.shape().to_vec()));
           }
           if a.shape().len() == 0 || (a.shape().len() == 1 && a.shape()[0] == 1) {
               let scalar = a.data()[0];
               let new_data: Vec<f32> = b.data().iter().map(|&x| x + scalar).collect();
               return Ok(Tensor::new(new_data, b.shape().to_vec()));
           }
           
           // TODO: implement full broadcasting
           return Err(anyhow::anyhow!("Add: shapes {:?} and {:?} mismatch (broadcasting not fully implemented)", a.shape(), b.shape()));
      }

      let new_data: Vec<f32> = a.data()
          .iter()
          .zip(b.data().iter())
          .map(|(&x, &y)| x + y)
          .collect();
      
      Ok(Tensor::new(new_data, a.shape().to_vec()))
  }
}