// https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
use neuroxyde::loader::ModelLoader;
use neuroxyde::graph::Graph;
use neuroxyde::runtime::InferenceSession;
use neuroxyde::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        panic!("Usage: mnist_infer <model_path>");
    };

    let model = ModelLoader::load_from_file(model_path)?;
    // std::fs::write("model.txt", format!("{:#?}", &model.model.graph))?;

    let graph = Graph::from_model(&model)?;
    let session = InferenceSession::new(graph)?;
    // println!("Model: {:#?}", model);
    // println!("{}", graph);
    let input = vec![Tensor::zeros(&[1, 1, 28, 28])];
    let outputs = session.run(&input)?;
    println!("Outputs: {:?}", outputs);
    Ok(())
}

