

if __name__ == "__main__":
    from ultralytics import YOLO
    import os
    import random
    import yaml
    import time
    import glob

    # Build a YOLOv9c model from scratch
    model = YOLO("yolov9c.yaml")
    model = YOLO("yolov9c.pt")

    # Display model information (optional)
    model.info()

    # Use cross-platform relative paths
    data_dir = os.path.join('content', 'data')
    images_dir = os.path.join(data_dir, 'images')

    # Get all image files in the images directory
    if os.path.exists(images_dir):
        image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    else:
        print(f"Images directory not found: {images_dir}")
        image_files = []

    # Define the training split ratio
    train_ratio = 0.8  # 80% for training, 20% for validation

    # Shuffle the image files
    random.shuffle(image_files)

    # Split the files into training and validation sets
    train_files = image_files[:int(len(image_files) * train_ratio)]
    val_files = image_files[int(len(image_files) * train_ratio):]

    # Write the training image paths to train.txt
    train_txt_path = os.path.join(data_dir, 'train.txt')
    with open(train_txt_path, 'w') as f:
        for img_path in train_files:
            f.write(img_path + '\n')

    # Write the validation image paths to val.txt
    val_txt_path = os.path.join(data_dir, 'val.txt')
    with open(val_txt_path, 'w') as f:
        for img_path in val_files:
            f.write(img_path + '\n')

    print(f"Automatically split {len(image_files)} images:")
    print(f"- {len(train_files)} images written to {train_txt_path}")
    print(f"- {len(val_files)} images written to {val_txt_path}")

    # Define the content of the data.yaml file with paths relative to data.yaml location
    data_yaml_content = {
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': len(open(os.path.join(data_dir, 'classes.txt')).readlines()),
        'names': [line.strip() for line in open(os.path.join(data_dir, 'classes.txt')).readlines()]
    }

    # Write the data.yaml file
    with open(os.path.join(data_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml_content, f)

    print(f"Created data.yaml at {os.path.join(data_dir, 'data.yaml')}")

    # Measure training time
    train_start = time.time()
    results = model.train(data=os.path.join(data_dir, "data.yaml"), epochs=100, imgsz=320, batch=16, device='0')
    train_end = time.time()
    train_time = train_end - train_start


    # Find the latest 'train*' directory under runs/detect and use its weights
    detect_dir = os.path.join('runs', 'detect')
    train_dirs = [d for d in glob.glob(os.path.join(detect_dir, 'train*')) if os.path.isdir(d)]
    if train_dirs:
        # Sort by modification time, newest last
        train_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_train = train_dirs[0]
        weights_dir = os.path.join(latest_train, 'weights')
        best_model_path = os.path.join(weights_dir, 'best.pt')
        last_model_path = os.path.join(weights_dir, 'last.pt')
        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            print(f"Loaded best.pt from: {best_model_path}")
        else:
            best_model = None
            print(f"No best.pt found in {weights_dir}")
        if os.path.exists(last_model_path):
            last_model = YOLO(last_model_path)
            print(f"Loaded last.pt from: {last_model_path}")
        else:
            last_model = None
            print(f"No last.pt found in {weights_dir}")
    else:
        best_model = None
        last_model = None
        print("No train* directory found under runs/detect.")


    # If you want to print training results summary
    print(results)

    test_data_dir = os.path.join('content','data', 'test_data')
    if os.path.exists(test_data_dir):
        image_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    else:
        print(f"Test data directory not found: {test_data_dir}")
        image_files = []

    # Measure inference times (only if best_model is available)
    inference_times = []
    if best_model is not None:
        for image_file in image_files:
            print(f"Running inference on {image_file}")
            inf_start = time.time()
            results = best_model(image_file)
            inf_end = time.time()
            inference_time = inf_end - inf_start
            inference_times.append(inference_time)
        if inference_times:
            avg_inference_time = sum(inference_times) / len(inference_times)
        else:
            avg_inference_time = 0
        print(f"\nTraining time: {train_time:.2f} seconds")
        print(f"Total inference time for {len(image_files)} images: {sum(inference_times):.2f} seconds")
        print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
    else:
        print("Skipping inference: No best.pt model available.")

