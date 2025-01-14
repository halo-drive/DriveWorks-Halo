import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging
import traceback
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ENGINE_PATH = "./yolov8n.trt"
INPUT_SHAPE = (640, 640)
CONF_THRESHOLD = 0.25
# Constants
COCO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

CLASS_NAMES = COCO_NAMES

class YOLOv8nDetector:
    def __init__(self, engine_path):
        logger.info("Initializing YOLOv8n-seg TensorRT detector...")
        self.logger = trt.Logger(trt.Logger.INFO)

        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Input/output shapes from engine bindings
        self.input_shape = (1, 3, 640, 640)
        self.output_shapes = [
            (1, 116, 8400),  # Detection output
            (1, 32, 160, 160)  # Segmentation output
        ]

        # Calculate total elements needed
        input_size = int(np.prod(self.input_shape))  # Convert to Python int
        output_sizes = [int(np.prod(shape)) for shape in self.output_shapes]

        # Allocate device memory
        self.d_input = cuda.mem_alloc(input_size * np.dtype(np.float32).itemsize)
        self.d_outputs = [cuda.mem_alloc(size * np.dtype(np.float32).itemsize)
                          for size in output_sizes]

        # Allocate host pinned memory with correct shape handling
        self.h_input = cuda.pagelocked_empty((input_size,), dtype=np.float32)
        self.h_outputs = [cuda.pagelocked_empty((size,), dtype=np.float32)
                          for size in output_sizes]

        logger.info("YOLOv8n TensorRT detector initialized successfully.")

    def _nms(self, boxes, scores, iou_threshold):
        """
        Perform Non-Maximum Suppression to filter overlapping bounding boxes.

        Args:
            boxes (np.ndarray): Array of shape (N, 4) containing bounding boxes coordinates [x1, y1, x2, y2]
            scores (np.ndarray): Array of shape (N,) containing confidence scores
            iou_threshold (float): IoU threshold for filtering overlapping boxes

        Returns:
            np.ndarray: Array of indices of kept boxes after NMS
        """
        # Ensure inputs are numpy arrays
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of each box
        areas = (x2 - x1) * (y2 - y1)

        # Sort boxes by scores (descending)
        order = scores.argsort()[::-1]

        keep = []  # Indices of kept boxes

        while order.size > 0:
            # Pick the box with highest score
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Calculate IoU with rest of boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # Calculate intersection area
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            # Calculate IoU
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]  # +1 because we're looking at order[1:]

        return keep

    def preprocess(self, image_path):
        """Preprocess image with diagnostic logging"""
        logger.info(f"Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Original image size: {image.size}")

        image_resized = image.resize(INPUT_SHAPE, Image.BICUBIC)
        image_array = np.asarray(image_resized, dtype=np.float32) / 255.0
        logger.info(f"Normalized input range: [{image_array.min():.3f}, {image_array.max():.3f}]")

        # YOLOv8 normalization verification
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        logger.info(f"Post-normalization range: [{image_array.min():.3f}, {image_array.max():.3f}]")

        image_transposed = np.transpose(image_array, (2, 0, 1))
        logger.info(f"Transposed shape: {image_transposed.shape}")
        return np.expand_dims(image_transposed, axis=0), image

    def postprocess(self, outputs, conf_threshold=0.15, iou_threshold=0.45):
        try:
            # Initial tensor analysis
            detections = outputs[0].reshape(1, 116, 8400).transpose(2, 1, 0).squeeze()
            logger.info(f"Reshaped detection tensor: {detections.shape}")

            # 1. Raw tensor statistics before any processing
            logger.info("\n=== Raw Tensor Analysis ===")
            logger.info(f"Detection value range: [{detections.min():.4f}, {detections.max():.4f}]")
            logger.info(f"Box coordinates range: [{detections[:, :4].min():.4f}, {detections[:, :4].max():.4f}]")
            logger.info(f"Raw objectness logits: [{detections[:, 4].min():.4f}, {detections[:, 4].max():.4f}]")

            # 2. Confidence computation analysis
            obj_conf = 1 / (1 + np.exp(-np.clip(detections[:, 4], -50, 50)))
            logger.info("\n=== Confidence Analysis ===")
            logger.info(f"Objectness confidence range: [{obj_conf.min():.4f}, {obj_conf.max():.4f}]")

            # 3. Class probability analysis
            class_logits = detections[:, 5:85]
            class_logits = class_logits - np.max(class_logits, axis=1, keepdims=True)
            class_scores = np.exp(class_logits)
            class_scores = class_scores / np.sum(class_scores, axis=1, keepdims=True)
            class_ids = np.argmax(class_scores, axis=1)
            class_conf = np.max(class_scores, axis=1)
            logger.info(f"Class confidence range: [{class_conf.min():.4f}, {class_conf.max():.4f}]")

            # 4. Combined score analysis
            scores = obj_conf * class_conf
            logger.info(f"Combined scores range: [{scores.min():.4f}, {scores.max():.4f}]")

            # 5. Pre-filtering statistics
            mask = scores > conf_threshold
            n_candidates = np.sum(mask)
            logger.info(f"\n=== Filtering Analysis ===")
            logger.info(f"Confidence threshold: {conf_threshold}")
            logger.info(f"Pre-NMS detections: {n_candidates}")

            if n_candidates > 0:
                boxes = boxes[mask]
                scores = scores[mask]
                class_ids = class_ids[mask]

                # Now safe to compute score range
                logger.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

                # Process coordinates only for valid detections
                boxes[:, 0] = np.clip(boxes[:, 0], 0, 1) * self.input_shape[1]
                boxes[:, 1] = np.clip(boxes[:, 1], 0, 1) * self.input_shape[0]
                boxes[:, 2] = np.clip(boxes[:, 2], 0, 1) * self.input_shape[1]
                boxes[:, 3] = np.clip(boxes[:, 3], 0, 1) * self.input_shape[0]

                # Convert to corner format
                x1 = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, self.input_shape[1])
                y1 = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, self.input_shape[0])
                x2 = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, self.input_shape[1])
                y2 = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, self.input_shape[0])
                boxes = np.stack([x1, y1, x2, y2], axis=1)

                keep = self._nms(boxes, scores, iou_threshold)
                return boxes[keep], scores[keep], class_ids[keep], outputs[1]

            return np.array([]), np.array([]), np.array([]), outputs[1]

        except Exception as e:
            logger.error(f"Postprocess error: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([]), np.array([]), np.array([]), None

    def infer(self, image_path):
        try:
            input_data, original_image = self.preprocess(image_path)
            np.copyto(self.h_input, input_data.ravel())

            # Transfer input to device
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

            # Run inference
            bindings = [int(self.d_input)] + [int(d_output) for d_output in self.d_outputs]
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

            # Transfer predictions back
            outputs = []
            for h_output, d_output in zip(self.h_outputs, self.d_outputs):
                cuda.memcpy_dtoh_async(h_output, d_output, self.stream)
                outputs.append(h_output)

            self.stream.synchronize()

            # Process detections
            boxes, scores, class_ids, segmentation = self.postprocess(outputs)

            # Return all five expected values
            return boxes, scores, class_ids, segmentation, original_image

        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            raise


def main(image_path):
    detector = YOLOv8nDetector(ENGINE_PATH)
    boxes, scores, class_ids, segmentation, image = detector.infer(image_path)

    logger.info("Detections:")
    for box, score, class_id in zip(boxes, scores, class_ids):
        # Convert box coordinates to integers for display
        box = box.astype(int)
        logger.info(f"Class: {CLASS_NAMES[class_id]} ({class_id}), "
                    f"Score: {score:.2f}, "
                    f"Box: {box.tolist()}")

        # Verification
        if class_id == 11:  # COCO class index for stop sign
            logger.info("Stop sign detected with high confidence")

if __name__ == "__main__":
    main("stopsign.jpg")  # Replace with your image path
