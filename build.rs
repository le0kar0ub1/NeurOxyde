fn main() -> std::io::Result<()> {
    // Check if onnx.proto3 exists before trying to compile it
    let proto_path = "onnx_proto/onnx.proto3";
    if std::path::Path::new(proto_path).exists() {
        let mut config = prost_build::Config::new();
        // Set output directory to src/onnx_generated/
        config.out_dir("src/onnx_generated/");
        
        config.compile_protos(&[proto_path], &["onnx_proto/"])?;
        println!("cargo:rerun-if-changed=onnx_proto/onnx.proto3");
    } else {
        println!("cargo:warning=ONNX proto file not found at {}, skipping generation", proto_path);
    }
    Ok(())
}

