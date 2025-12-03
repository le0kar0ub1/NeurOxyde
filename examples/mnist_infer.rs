// https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
use neuroxyde::loader::ModelLoader;
use neuroxyde::graph::Graph;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        panic!("Usage: mnist_infer <model_path>");
    };

    let model = ModelLoader::load_from_file(model_path)?;
    let graph = Graph::from_model(&model)?;
    // println!("Model: {:#?}", model);
    println!("{}", graph);
    Ok(())
}

