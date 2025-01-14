from PIL import Image
import numpy as np

# Configuration
input_jpg_path = "stopsign.jpg"  # Path to your input JPG file
output_raw_path = "input_tensor.raw"  # Path to save the binary raw file
input_shape = (1, 3, 640, 640)  # Model input shape (batch_size, channels, height, width)

# Load and preprocess the image
image = Image.open(input_jpg_path).convert('RGB')  # Convert to RGB
image = image.resize((input_shape[2], input_shape[3]))  # Resize to 640x640

# Convert to numpy array and normalize pixel values
image_np = np.array(image).astype(np.float32) / 255.0  # Scale pixel values to [0, 1]

# Convert to NCHW format (channels first)
image_np = np.transpose(image_np, (2, 0, 1))  # Convert from HWC to CHW

# Add batch dimension
image_np = np.expand_dims(image_np, axis=0)  # Add batch size dimension

# Save to a raw binary file
image_np.tofile(output_raw_path)
print(f"Preprocessed input saved to: {output_raw_path}")
