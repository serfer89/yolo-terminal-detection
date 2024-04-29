#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import cv2
import supervision as sv
from super_gradients.training import models

# Define default values
DEFAULT_CHECKPOINT_PATH = "/home/serg/Documents/yolo-nas/models/yolo_nas_l(250).pth"
DEFAULT_IMAGE_PATH = "/media/serg/CEE6C5BCE6C5A555/Users/pavlo/Documents/ds100/dataset100/1704713877.jpeg"
DEFAULT_CONFIDENCE_THRESHOLD = 0.56
DEFAULT_MODEL_ARCH = 'yolo_nas_l'
DEFAULT_LOCATION = "/home/serg/Documents/yolo-nas/wac-2-1"
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

# Define available classes
CLASSES = {
    'person': 11,
    'car': 636,
    'air-conditioner': 2119,
    'window': 5391
}
CLASSES = sorted(CLASSES.keys())

def main(args):
    checkpoint_path = args.checkpoint_path
    image_path = args.image_path
    confidence_threshold = args.confidence_threshold
    model_arch = args.model_arch
    location = args.location
    device = args.device
    
    dataset_params = {
        'data_dir': location,
        'train_images_dir':'train/images',
        'train_labels_dir':'train/labels',
        'val_images_dir':'valid/images',
        'val_labels_dir':'valid/labels',
        'test_images_dir':'test/images',
        'test_labels_dir':'test/labels',
        'classes': CLASSES
    }

    best_model = models.get(model_arch, num_classes=len(dataset_params['classes']),
                            checkpoint_path=checkpoint_path
                           ).to(device)

    image = cv2.imread(image_path)
    result = best_model.predict(image, conf=confidence_threshold)
    detections = sv.Detections.from_yolo_nas(result)
    box_annotator = sv.BoxAnnotator()

    labels = [
        f"{CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = box_annotator.annotate(
        image.copy(), detections=detections, labels=labels
    )

    #sv.plot_image(image=annotated_image, size=(8, 8))

    # Count found items per class
    found_items = {class_name: 0 for class_name in CLASSES}
    for class_id in detections.class_id:
        class_name = CLASSES[class_id]
        found_items[class_name] += 1

    print("Found items per class:")
    for class_name, count in found_items.items():
        print(f"{class_name}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO NAS Detection Script")
    parser.add_argument("-image_path", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to the image for detection")
    parser.add_argument("-checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help="Path to the checkpoint file for the model")
    parser.add_argument("-confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for detection")
    parser.add_argument("-model_arch", type=str, default=DEFAULT_MODEL_ARCH,
                        help="Model architecture")
    parser.add_argument("-location", type=str, default=DEFAULT_LOCATION,
                        help="Location of the dataset")
    parser.add_argument("-device", type=str, default=DEFAULT_DEVICE,
                        help="Device to run the model on (cpu or cuda)")
    args = parser.parse_args()
    main(args)

