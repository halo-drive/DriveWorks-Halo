import onnx

# Load the ONNX model
model_path = "yolov8x-seg.onnx"  # Replace with your ONNX model path
model = onnx.load(model_path)

# Check inputs
print("Model Inputs:")
for input_tensor in model.graph.input:
    print(f"Name: {input_tensor.name}")
    tensor_type = input_tensor.type.tensor_type
    if tensor_type.HasField("shape"):
        shape = [dim.dim_value if dim.HasField("dim_value") else "dynamic" for dim in tensor_type.shape.dim]
        print(f"Shape: {shape}")
    print(f"Data Type: {onnx.TensorProto.DataType.Name(tensor_type.elem_type)}")

# Check outputs
print("\nModel Outputs:")
for output_tensor in model.graph.output:
    print(f"Name: {output_tensor.name}")
    tensor_type = output_tensor.type.tensor_type
    if tensor_type.HasField("shape"):
        shape = [dim.dim_value if dim.HasField("dim_value") else "dynamic" for dim in tensor_type.shape.dim]
        print(f"Shape: {shape}")
    print(f"Data Type: {onnx.TensorProto.DataType.Name(tensor_type.elem_type)}")

# Check initializers
print("\nModel Initializers (Weights/Biases):")
for initializer in model.graph.initializer:
    print(f"Name: {initializer.name}, DataType: {onnx.TensorProto.DataType.Name(initializer.data_type)}, Shape: {list(initializer.dims)}")
