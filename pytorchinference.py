import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Configure the device with error handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(model_path):
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def preprocess_image(image_path):
    try:
        # Load image in RGB format
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully from: {image_path}")
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise


def postprocess(results):
    try:
        processed_results = []
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for box in boxes:
                processed_results.append({
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].cpu().numpy()  # Convert to numpy array
                })
        return processed_results
    except Exception as e:
        print(f"Error in postprocessing: {str(e)}")
        raise


def draw_detections(image, detections):
    try:
        image_copy = image.copy()
        for det in detections:
            bbox = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"

            # Convert coordinates to integers
            xmin, ymin, xmax, ymax = map(int, bbox)

            # Draw bounding box
            cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Add label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image_copy, (xmin, ymin - 20), (xmin + label_size[0], ymin), (0, 255, 0), -1)
            cv2.putText(image_copy, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image_copy
    except Exception as e:
        print(f"Error drawing detections: {str(e)}")
        raise


def main(model_path, image_path, class_names, conf_threshold=0.25):
    try:
        # Load model
        model = load_model(model_path)

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Run inference
        results = model(image, conf=conf_threshold)

        # Process results
        detections = postprocess(results)

        # Draw detections
        annotated_image = draw_detections(image, detections)

        # Display results
        cv2.imshow("Inference Results", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Detected {len(detections)} objects")
        for det in detections:
            print(f"Class: {det['class']}, Confidence: {det['confidence']:.2f}")

    except Exception as e:
        print(f"Error in main pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Inference with PyTorch")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classes", type=str, required=True, help="Path to class names file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    try:
        # Load class names
        with open(args.classes, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        # Run inference pipeline
        main(args.model, args.image, class_names, args.conf)

    except Exception as e:
        print(f"Critical error: {str(e)}")
        exit(1)