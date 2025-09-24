import logging
from ultralytics import YOLO
import os
import random
import yaml
import time
import glob


def train_model(
    base_model: str = "yolov9c",
    train_dataset: str = "20250924-100-augmented",
    train_ratio: float = 0.8,
    epochs: int = 100,
    img_size: int = 320,
    batch_size: int = 16,
    device: str = "cpu",
) -> None:
    # Build a YOLOv9c model from scratch
    model = YOLO(f"{base_model}.yaml")
    model = YOLO(f"{base_model}.pt")

    # Display model information (optional)
    model.info()

    # Use correct project-relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, "data", "train", train_dataset)
    images_dir = os.path.join(data_dir, "images")

    # Get all image files in the images directory
    if os.path.exists(images_dir):
        image_files = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
    else:
        print(f"Images directory not found: {images_dir}")
        image_files = []

    # Define the training split ratio
    train_ratio = 0.8  # 80% for training, 20% for validation

    # Shuffle the image files
    random.shuffle(image_files)

    # Split the files into training and validation sets
    train_files = image_files[: int(len(image_files) * train_ratio)]
    val_files = image_files[int(len(image_files) * train_ratio) :]

    # Write the training image paths to train.txt
    train_txt_path = os.path.join(data_dir, "train.txt")
    with open(train_txt_path, "w") as f:
        for img_path in train_files:
            f.write(img_path + "\n")

    # Write the validation image paths to val.txt
    val_txt_path = os.path.join(data_dir, "val.txt")
    with open(val_txt_path, "w") as f:
        for img_path in val_files:
            f.write(img_path + "\n")

    print(f"Automatically split {len(image_files)} images:")
    print(f"- {len(train_files)} images written to {train_txt_path}")
    print(f"- {len(val_files)} images written to {val_txt_path}")

    # Define the content of the data.yaml file with paths relative to data.yaml location
    data_yaml_content = {
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(open(os.path.join(data_dir, "classes.txt")).readlines()),
        "names": [
            line.strip()
            for line in open(os.path.join(data_dir, "classes.txt")).readlines()
        ],
    }

    # Write the data.yaml file
    with open(os.path.join(data_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml_content, f)

    print(f"Created data.yaml at {os.path.join(data_dir, 'data.yaml')}")

    # Measure training time
    train_start = time.time()
    results = model.train(
        data=os.path.join(data_dir, "data.yaml"),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
    )
    train_end = time.time()
    train_time = train_end - train_start
    logging.info(f"Training completed in {train_time:.2f} seconds.")
