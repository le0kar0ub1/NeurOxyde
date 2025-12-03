//! Tensor module: defines the Tensor struct and basic tensor utilities.
use crate::onnx::onnx_proto;

pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product::<usize>();
        Self { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    pub fn from_proto(tns: &onnx_proto::TensorProto) -> Self {
        use onnx_proto::tensor_proto::DataType;
    
        let shape: Vec<usize> = tns.dims.iter().map(|d| *d as usize).collect();
        let count = shape.iter().product::<usize>();
    
        match tns.data_type {
            x if x == DataType::Float as i32 || x == 1 => {
                // common: raw_data or float_data
                if !tns.raw_data.is_empty() {
                    let floats = tns.raw_data
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<f32>>();
                    assert_eq!(floats.len(), count);
                    return Tensor::new(floats, shape);
                }
                if !tns.float_data.is_empty() {
                    let v = tns.float_data.clone();
                    assert_eq!(v.len(), count);
                    return Tensor::new(v, shape);
                }
                panic!("No float data in TensorProto");
            }
            x if x == DataType::Double as i32 || x == 11 => {
                // convert f64 to f32
                if !tns.raw_data.is_empty() {
                    let doubles = tns.raw_data
                        .chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
                        .collect::<Vec<f32>>();
                    assert_eq!(doubles.len(), count);
                    return Tensor::new(doubles, shape);
                }
                // prost has `double_data` field named `double_data`? Many models store raw.
                panic!("Double TensorProto: only raw_data handling implemented");
            }
            x if x == DataType::Int32 as i32 || x == 6 => {
                if !tns.raw_data.is_empty() {
                    let ints = tns.raw_data
                        .chunks_exact(4)
                        .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
                        .collect::<Vec<f32>>();
                    assert_eq!(ints.len(), count);
                    return Tensor::new(ints, shape);
                }
                if !tns.int32_data.is_empty() {
                    let v = tns.int32_data.iter().map(|&i| i as f32).collect::<Vec<f32>>();
                    return Tensor::new(v, shape);
                }
                panic!("No int32 data found");
            }
            x if x == DataType::Int64 as i32 || x == 7 => {
                if !tns.raw_data.is_empty() {
                    let ints = tns.raw_data
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32)
                        .collect::<Vec<f32>>();
                    assert_eq!(ints.len(), count);
                    return Tensor::new(ints, shape);
                }
                if !tns.int64_data.is_empty() {
                    let v = tns.int64_data.iter().map(|&i| i as f32).collect::<Vec<f32>>();
                    return Tensor::new(v, shape);
                }
                panic!("No int64 data found");
            }
            _ => {
                panic!("Unsupported TensorProto data_type: {:?}", tns.data_type);
            }
        }
    }
}