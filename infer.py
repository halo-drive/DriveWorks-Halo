import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Constants
ENGINE_PATH = "yolov8x-seg.engine"
INPUT_SHAPE = (640, 640)

# COCO class names
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']


def preprocess_image(image, input_shape):
    """Preprocess image for model input"""
    image_resized = cv2.resize(image, input_shape)
    image_preprocessed = np.ascontiguousarray(image_resized.astype(np.float32) / 255.0)
    return np.expand_dims(image_preprocessed.transpose(2, 0, 1), axis=0)


def process_output(output, conf_threshold=0.25):
    """Process network output"""
    outputs = output.reshape((1, 116, 8400))[0].T

    # Get boxes, scores, and class predictions
    boxes = outputs[:, :4]
    scores = np.max(outputs[:, 4:84], axis=1)
    class_ids = np.argmax(outputs[:, 4:84], axis=1)

    # Filter by confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) > 0:
        # Convert from center format to corner format
        x = boxes[:, 0] / INPUT_SHAPE[0]
        y = boxes[:, 1] / INPUT_SHAPE[1]
        w = boxes[:, 2] / INPUT_SHAPE[0]
        h = boxes[:, 3] / INPUT_SHAPE[1]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes = np.clip(boxes, 0, 1)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.45)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
            return boxes, scores, class_ids

    return np.array([]), np.array([]), np.array([])


def draw_detections(image, boxes, scores, class_ids):
    """Draw detections on image"""
    h, w = image.shape[:2]

    for box, score, class_id in zip(boxes, scores, class_ids):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)

        # Generate color for class
        color = tuple(map(int, np.random.randint(100, 255, size=3)))

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add label
        label = f'{CLASS_NAMES[int(class_id)]}: {score:.2f}'
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


class YOLODetector:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Get input and output tensor details
        self.input_idx = self.engine.get_tensor_name(0)
        self.output0_idx = self.engine.get_tensor_name(1)
        self.output1_idx = self.engine.get_tensor_name(2)

        # Get shapes and create buffers
        self.init_buffers()

        # Set tensor addresses
        self.context.set_tensor_address(self.input_idx, int(self.d_input))
        self.context.set_tensor_address(self.output0_idx, int(self.d_output0))
        self.context.set_tensor_address(self.output1_idx, int(self.d_output1))

    def init_buffers(self):
        """Initialize host and device buffers"""
        # Get shapes
        input_shape = self.engine.get_tensor_shape(self.input_idx)
        output0_shape = self.engine.get_tensor_shape(self.output0_idx)
        output1_shape = self.engine.get_tensor_shape(self.output1_idx)

        # Calculate sizes
        self.input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
        self.output0_size = trt.volume(output0_shape) * np.dtype(np.float32).itemsize
        self.output1_size = trt.volume(output1_shape) * np.dtype(np.float32).itemsize

        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output0 = cuda.mem_alloc(self.output0_size)
        self.d_output1 = cuda.mem_alloc(self.output1_size)

        # Allocate host memory
        self.h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        self.h_output0 = cuda.pagelocked_empty(trt.volume(output0_shape), dtype=np.float32)
        self.h_output1 = cuda.pagelocked_empty(trt.volume(output1_shape), dtype=np.float32)

    def infer(self, image):
        # Preprocess input image
        input_data = preprocess_image(image, INPUT_SHAPE)
        np.copyto(self.h_input, input_data.ravel())

        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.h_output0, self.d_output0, self.stream)
        cuda.memcpy_dtoh_async(self.h_output1, self.d_output1, self.stream)

        # Synchronize stream
        self.stream.synchronize()

        # Process output
        return process_output(self.h_output0)

    def __del__(self):
        try:
            # Clean up resources
            del self.context
            del self.engine
            self.d_input.free()
            self.d_output0.free()
            self.d_output1.free()
        except:
            pass


def main():
    # Initialize detector
    detector = YOLODetector(ENGINE_PATH)

    # Open video capture
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            boxes, scores, class_ids = detector.infer(frame)

            # Draw detections
            frame = draw_detections(frame, boxes, scores, class_ids)

            # Show result
            cv2.namedWindow('TensorRT Detection on RTX 3060', cv2.WINDOW_FULLSCREEN)  # Allows the window to be resized
            frame_resized = cv2.resize(frame, (1200,720)) # Set the window to 1280x720 or any desired size

            cv2.imshow('TensorRT Detection on RTX 3060', frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        del detector


if __name__ == '__main__':
    main()