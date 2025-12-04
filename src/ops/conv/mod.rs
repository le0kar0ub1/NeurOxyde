//! Conv operator implementation (2D convolution)

use crate::ops::operator::Operator;
use crate::tensor::Tensor;
use crate::onnx::onnx_proto::NodeProto;

pub struct Conv;

impl Conv {
    /// Extract an i64 attribute from the node
    fn get_int_attr(node: &NodeProto, name: &str, default: i64) -> i64 {
        node.attribute.iter()
            .find(|a| a.name == name)
            .map(|a| a.i)
            .unwrap_or(default)
    }

    /// Extract a list of i64 attributes from the node
    fn get_ints_attr(node: &NodeProto, name: &str) -> Vec<i64> {
        node.attribute.iter()
            .find(|a| a.name == name)
            .map(|a| a.ints.clone())
            .unwrap_or_default()
    }

    /// Extract a string attribute from the node
    fn get_string_attr(node: &NodeProto, name: &str) -> String {
        node.attribute.iter()
            .find(|a| a.name == name)
            .map(|a| String::from_utf8_lossy(&a.s).to_string())
            .unwrap_or_default()
    }

    /// Compute padding based on auto_pad setting
    fn compute_auto_pad(
        auto_pad: &str,
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> (usize, usize, usize, usize) {
        // Returns (pad_top, pad_left, pad_bottom, pad_right)
        match auto_pad {
            "SAME_UPPER" | "SAME_LOWER" => {
                let effective_kh = (kernel_h - 1) * dilation_h + 1;
                let effective_kw = (kernel_w - 1) * dilation_w + 1;

                let out_h = (input_h + stride_h - 1) / stride_h;
                let out_w = (input_w + stride_w - 1) / stride_w;

                let pad_h = ((out_h - 1) * stride_h + effective_kh).saturating_sub(input_h);
                let pad_w = ((out_w - 1) * stride_w + effective_kw).saturating_sub(input_w);

                if auto_pad == "SAME_UPPER" {
                    (pad_h / 2, pad_w / 2, (pad_h + 1) / 2, (pad_w + 1) / 2)
                } else {
                    ((pad_h + 1) / 2, (pad_w + 1) / 2, pad_h / 2, pad_w / 2)
                }
            }
            "VALID" => (0, 0, 0, 0),
            _ => (0, 0, 0, 0), // NOTSET or unknown
        }
    }
}

impl Operator for Conv {
    fn run(&self, inputs: &[&Tensor], node: &NodeProto) -> anyhow::Result<Tensor> {
        // Input X: (N, C, H, W)
        // Weight W: (M, C/group, kH, kW)
        // Optional Bias B: (M,)
        let x = inputs[0];
        let w = inputs[1];
        let bias = inputs.get(2).copied();

        let x_shape = x.shape();
        let w_shape = w.shape();

        // println!("Conv: x_shape: {:?}, w_shape: {:?}", x_shape, w_shape);
        // println!("X data: {:?}", x.data());
        // println!("W data: {:?}", w.data());
        // println!("Bias data: {:?}", bias.map(|b| b.data()));

        if x_shape.len() != 4 || w_shape.len() != 4 {
            return Err(anyhow::anyhow!("Conv: only 2D convolution (4D tensors) supported"));
        }

        let batch = x_shape[0];
        let in_channels = x_shape[1];
        let in_h = x_shape[2];
        let in_w = x_shape[3];

        let out_channels = w_shape[0];
        let kernel_h = w_shape[2];
        let kernel_w = w_shape[3];

        // Parse attributes
        let group = Self::get_int_attr(node, "group", 1) as usize;
        let strides = Self::get_ints_attr(node, "strides");
        let dilations = Self::get_ints_attr(node, "dilations");
        let pads = Self::get_ints_attr(node, "pads");
        let auto_pad = Self::get_string_attr(node, "auto_pad");

        let stride_h = strides.first().copied().unwrap_or(1) as usize;
        let stride_w = strides.get(1).copied().unwrap_or(stride_h as i64) as usize;

        let dilation_h = dilations.first().copied().unwrap_or(1) as usize;
        let dilation_w = dilations.get(1).copied().unwrap_or(dilation_h as i64) as usize;

        // Compute padding
        let (pad_top, pad_left, pad_bottom, pad_right) = if !auto_pad.is_empty() && auto_pad != "NOTSET" {
            Self::compute_auto_pad(&auto_pad, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w)
        } else if pads.len() >= 4 {
            (pads[0] as usize, pads[1] as usize, pads[2] as usize, pads[3] as usize)
        } else {
            (0, 0, 0, 0)
        };

        // Compute output dimensions
        let effective_kh = (kernel_h - 1) * dilation_h + 1;
        let effective_kw = (kernel_w - 1) * dilation_w + 1;
        let out_h = (in_h + pad_top + pad_bottom - effective_kh) / stride_h + 1;
        let out_w = (in_w + pad_left + pad_right - effective_kw) / stride_w + 1;

        // Allocate output
        let mut output = vec![0.0f32; batch * out_channels * out_h * out_w];

        let x_data = x.data();
        let w_data = w.data();

        let in_channels_per_group = in_channels / group;
        let out_channels_per_group = out_channels / group;

        // Perform convolution
        for n in 0..batch {
            for g in 0..group {
                for oc in 0..out_channels_per_group {
                    let abs_oc = g * out_channels_per_group + oc;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0f32;

                            for ic in 0..in_channels_per_group {
                                let abs_ic = g * in_channels_per_group + ic;
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let ih = (oh * stride_h + kh * dilation_h) as isize - pad_top as isize;
                                        let iw = (ow * stride_w + kw * dilation_w) as isize - pad_left as isize;

                                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                            let ih = ih as usize;
                                            let iw = iw as usize;

                                            let x_idx = n * in_channels * in_h * in_w
                                                + abs_ic * in_h * in_w
                                                + ih * in_w
                                                + iw;

                                            let w_idx = abs_oc * (in_channels_per_group * kernel_h * kernel_w)
                                                + ic * kernel_h * kernel_w
                                                + kh * kernel_w
                                                + kw;

                                            sum += x_data[x_idx] * w_data[w_idx];
                                        }
                                    }
                                }
                            }

                            // Add bias if present
                            if let Some(b) = bias {
                                sum += b.data()[abs_oc];
                            }

                            let out_idx = n * out_channels * out_h * out_w
                                + abs_oc * out_h * out_w
                                + oh * out_w
                                + ow;
                            output[out_idx] = sum;
                        }
                    }
                }
            }
        }

        Ok(Tensor::new(output, vec![batch, out_channels, out_h, out_w]))
    }
}

