import onnx

# Load the ONNX model
model_path = "simswap_pruned.onnx"
onnx_model = onnx.load(model_path)

# Check that the model is well-formed
onnx.checker.check_model(onnx_model)

# Print a human-readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))