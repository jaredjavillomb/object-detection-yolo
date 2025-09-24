
import os
import glob
import yaml
from ultralytics import YOLO
from pathlib import Path
import logging


def create_validation_yaml(validate_folder_name:str = '20250924-20-real')->str:
    """
    Create a validation-specific data.yaml file pointing to a chosen validation folder inside data/validate.
    Args:
        validate_folder_name (str): Name of the validation folder (e.g., '20250924-20-real').
    """
    # Compose the full path to the validation folder
    validate_folder = os.path.join('data', 'validate', validate_folder_name)
    # Read the classes from the chosen validation folder
    classes_path = os.path.join(validate_folder, 'classes.txt')
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    # Root path is the parent of the validation folder
    root_path = os.path.abspath(os.path.dirname(validate_folder))
    val_images = os.path.join(validate_folder_name, 'images')

    # Create validation data configuration
    val_data_config = {
        'path': root_path,  # Root path
        'train': 'data/train.txt',  # Keep original training data
        'val': val_images,  # Point to validation images
        'test': val_images,  # Use same folder for testing
        'nc': len(class_names),
        'names': class_names
    }

    # Save validation yaml inside the chosen validate folder
    val_yaml_path = os.path.join(validate_folder, 'validation_data.yaml')
    with open(val_yaml_path, 'w') as f:
        yaml.dump(val_data_config, f, default_flow_style=False)


    logging.info(f"Created validation data config: {val_yaml_path}")
    logging.info(f"Classes found: {class_names}")
    logging.info(f"Number of classes: {len(class_names)}")

    return val_yaml_path

def parse_yolo_label(label_path:str)->list:
    """
    Parse YOLO format label file to extract ground truth bounding boxes and classes
    """
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        boxes.append({
                            'class': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
    return boxes


def validate_yolo_model(model_id:str="", validate_dataset:str="")-> None:
    """
    Validate YOLO model on valid-data dataset and calculate comprehensive metrics
    Args:
        model_id (str): Model identifier (e.g., '20250924') or path to the YOLO model (.pt file).
    """

    logging.info("="*60)
    logging.info("YOLO MODEL VALIDATION ON VALID-DATA DATASET")
    logging.info("="*60)

    # If model_id is a .pt file, use as is; otherwise, treat as model folder name

    if not model_id:
        logging.error("Please provide a model_id (e.g., '20250924') or a path to a YOLO .pt file.")
        return
    if model_id.endswith('.pt'):
        model_path = model_id
    else:
        model_path = os.path.join('models', model_id, 'best.pt')

    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        return

    # Load the specified model
    model = YOLO(model_path)
    logging.info(f"Loaded model from: {model_path}")
    best_model_path = model_path

    validate_folder = os.path.join('data', 'validate', validate_dataset)

    # Create validation yaml configuration
    val_yaml_path = create_validation_yaml(validate_dataset)

    # Define validation dataset paths
    valid_images_dir = os.path.join(validate_folder, 'images')
    valid_labels_dir = os.path.join(validate_folder, 'labels')

    # Check if validation data exists
    if not os.path.exists(valid_images_dir):
        logging.error(f"Validation images directory not found: {valid_images_dir}")
        return

    if not os.path.exists(valid_labels_dir):
        logging.error(f"Validation labels directory not found: {valid_labels_dir}")
        return

    # Get validation images
    valid_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        valid_images.extend(glob.glob(os.path.join(valid_images_dir, ext)))
        valid_images.extend(glob.glob(os.path.join(valid_images_dir, ext.upper())))

    if not valid_images:
        logging.error("No validation images found.")
        return

    logging.info(f"Found {len(valid_images)} validation images")
    logging.info(f"Expected {len(valid_images)} corresponding label files")

    # Verify label files exist
    label_count = 0
    for img_path in valid_images:
        img_name = Path(img_path).stem
        label_path = os.path.join(valid_labels_dir, f"{img_name}.txt")
        if os.path.exists(label_path):
            label_count += 1

    logging.info(f"Found {label_count} label files")

    # Run official YOLO validation with ground truth
    logging.info("="*50)
    logging.info("RUNNING OFFICIAL YOLO VALIDATION WITH GROUND TRUTH")
    logging.info("="*50)

    try:
        # Run validation using the validation yaml
        val_results = model.val(
            data=val_yaml_path,
            split='val',
            save_json=True,
            save_hybrid=True,
            plots=True,
            verbose=True
        )

        logging.info(f"OFFICIAL VALIDATION RESULTS:")
        logging.info(f"{'Metric':<20} {'Value':<10}")
        logging.info("-" * 30)
        logging.info(f"{'mAP@0.5':<20} {val_results.box.map50:.4f}")
        logging.info(f"{'mAP@0.5:0.95':<20} {val_results.box.map:.4f}")
        logging.info(f"{'Precision':<20} {val_results.box.mp:.4f}")
        logging.info(f"{'Recall':<20} {val_results.box.mr:.4f}")
        if hasattr(val_results.box, 'maps') and hasattr(val_results.box, 'p') and hasattr(val_results.box, 'r'):
            logging.info("CLASS-WISE RESULTS:")
            logging.info(f"{'Class':<20} {'mAP@0.5:0.95':<15} {'Precision':<10} {'Recall':<10}")
            logging.info("-" * 60)
            for i, class_name in enumerate(val_results.names):
                map_class = val_results.box.maps[i] if i < len(val_results.box.maps) else float('nan')
                precision = val_results.box.p[i] if i < len(val_results.box.p) else float('nan')
                recall = val_results.box.r[i] if i < len(val_results.box.r) else float('nan')
                logging.info(f"{class_name:<20} {map_class:.4f}      {precision:.4f}     {recall:.4f}")
    except Exception as e:
        logging.error(f"Error during YOLO validation: {e}")
        return


if __name__ == "__main__":
    validate_yolo_model()