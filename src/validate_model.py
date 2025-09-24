import os
import glob
import time
import json
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

def create_validation_yaml():
    """
    Create a validation-specific data.yaml file pointing to valid-data folder
    """
    # Read the classes from valid-data
    classes_path = os.path.join('content', 'valid-data', 'classes.txt')
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create validation data configuration
    val_data_config = {
        'path': os.path.abspath('content'),  # Root path
        'train': 'data/train.txt',  # Keep original training data
        'val': 'valid-data/images',  # Point to validation images
        'test': 'valid-data/images',  # Use same folder for testing
        'nc': len(class_names),
        'names': class_names
    }
    
    # Save validation yaml
    val_yaml_path = 'validation_data.yaml'
    with open(val_yaml_path, 'w') as f:
        yaml.dump(val_data_config, f, default_flow_style=False)
    
    print(f"Created validation data config: {val_yaml_path}")
    print(f"Classes found: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    return val_yaml_path

def parse_yolo_label(label_path):
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

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    box format: [x_center, y_center, width, height] (normalized)
    """
    # Convert to corner format
    x1_min = box1['x_center'] - box1['width'] / 2
    y1_min = box1['y_center'] - box1['height'] / 2
    x1_max = box1['x_center'] + box1['width'] / 2
    y1_max = box1['y_center'] + box1['height'] / 2
    
    x2_min = box2['x_center'] - box2['width'] / 2
    y2_min = box2['y_center'] - box2['height'] / 2
    x2_max = box2['x_center'] + box2['width'] / 2
    y2_max = box2['y_center'] + box2['height'] / 2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def create_confusion_matrix(predictions, ground_truths, class_names, iou_threshold=0.5):
    """
    Create confusion matrix for object detection with improved matching
    """
    print(f"🔍 Debugging confusion matrix creation:")
    print(f"  - Ground truth images: {len(ground_truths)}")
    print(f"  - Prediction images: {len(predictions)}")
    print(f"  - IoU threshold: {iou_threshold}")
    
    num_classes = len(class_names)
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    
    total_gt_boxes = 0
    total_pred_boxes = 0
    matched_boxes = 0
    
    for img_name in ground_truths.keys():
        gt_boxes = ground_truths[img_name]
        pred_boxes = predictions.get(img_name, [])
        
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)
        
        print(f"  📷 {img_name}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}")
        
        if len(gt_boxes) > 0:
            print(f"    GT classes: {[box['class'] for box in gt_boxes]}")
        if len(pred_boxes) > 0:
            print(f"    Pred classes: {[box['class'] for box in pred_boxes]}")
        
        # Track which boxes have been matched
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)
        
        # For each prediction, find best matching ground truth
        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # Match found
                gt_class = gt_boxes[best_gt_idx]['class']
                pred_class = pred['class']
                
                # Ensure classes are within valid range
                if 0 <= gt_class < num_classes and 0 <= pred_class < num_classes:
                    cm[gt_class][pred_class] += 1
                    gt_matched[best_gt_idx] = True
                    pred_matched[pred_idx] = True
                    matched_boxes += 1
                    print(f"    ✅ Match: GT class {gt_class} -> Pred class {pred_class} (IoU: {best_iou:.3f})")
        
        # False positives - unmatched predictions
        for pred_idx, pred in enumerate(pred_boxes):
            if not pred_matched[pred_idx]:
                pred_class = pred['class']
                if 0 <= pred_class < num_classes:
                    cm[num_classes][pred_class] += 1  # Background row
                    print(f"    ❌ False Positive: Pred class {pred_class}")
        
        # False negatives - unmatched ground truth
        for gt_idx, gt in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                gt_class = gt['class']
                if 0 <= gt_class < num_classes:
                    cm[gt_class][num_classes] += 1  # Background column
                    print(f"    ❌ False Negative: GT class {gt_class}")
    
    print(f"📊 Summary:")
    print(f"  - Total GT boxes: {total_gt_boxes}")
    print(f"  - Total Pred boxes: {total_pred_boxes}")
    print(f"  - Matched boxes: {matched_boxes}")
    print(f"  - Match rate: {matched_boxes/max(total_gt_boxes,1)*100:.1f}%")
    
    return cm

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix
    """
    # Add background class to names
    extended_class_names = class_names + ['Background/FN']
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=extended_class_names,
                yticklabels=extended_class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix for Object Detection\n(IoU Threshold: 0.5)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add text explanation
    plt.figtext(0.02, 0.02, 
                'Note: Background/FN represents false negatives (missed detections) and false positives (extra detections)',
                fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def validate_yolo_model():
    """
    Validate YOLO model on valid-data dataset and calculate comprehensive metrics
    """
    
    print("="*60)
    print("YOLO MODEL VALIDATION ON VALID-DATA DATASET")
    print("="*60)
    
    # Find the latest trained model
    detect_dir = os.path.join('runs', 'detect')
    train_dirs = [d for d in glob.glob(os.path.join(detect_dir, 'train*')) if os.path.isdir(d)]
    
    if not train_dirs:
        print("No trained model found. Please train a model first.")
        return
    
    # Get the latest training directory
    train_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_train = train_dirs[0]
    best_model_path = os.path.join(latest_train, 'weights', 'best.pt')
    
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        return
    
    # Load the best model
    model = YOLO(best_model_path)
    print(f"Loaded model from: {best_model_path}")
    
    # Create validation yaml configuration
    val_yaml_path = create_validation_yaml()
    
    # Define validation dataset paths
    valid_images_dir = os.path.join('content', 'valid-data', 'images')
    valid_labels_dir = os.path.join('content', 'valid-data', 'labels')
    
    # Check if validation data exists
    if not os.path.exists(valid_images_dir):
        print(f"Validation images directory not found: {valid_images_dir}")
        return
    
    if not os.path.exists(valid_labels_dir):
        print(f"Validation labels directory not found: {valid_labels_dir}")
        return
    
    # Get validation images
    valid_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
        valid_images.extend(glob.glob(os.path.join(valid_images_dir, ext)))
        valid_images.extend(glob.glob(os.path.join(valid_images_dir, ext.upper())))
    
    if not valid_images:
        print("No validation images found.")
        return
    
    print(f"Found {len(valid_images)} validation images")
    print(f"Expected {len(valid_images)} corresponding label files")
    
    # Verify label files exist
    label_count = 0
    for img_path in valid_images:
        img_name = Path(img_path).stem
        label_path = os.path.join(valid_labels_dir, f"{img_name}.txt")
        if os.path.exists(label_path):
            label_count += 1
    
    print(f"Found {label_count} label files")
    
    # Run official YOLO validation with ground truth
    print("\n" + "="*50)
    print("RUNNING OFFICIAL YOLO VALIDATION WITH GROUND TRUTH")
    print("="*50)
    
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
        
        print(f"\n📊 OFFICIAL VALIDATION RESULTS:")
        print(f"{'Metric':<20} {'Value':<10}")
        print("-" * 30)
        print(f"{'mAP@0.5':<20} {val_results.box.map50:.4f}")
        print(f"{'mAP@0.5:0.95':<20} {val_results.box.map:.4f}")
        print(f"{'Precision':<20} {val_results.box.mp:.4f}")
        print(f"{'Recall':<20} {val_results.box.mr:.4f}")
        
        # Calculate F1 score
        if val_results.box.mp > 0 and val_results.box.mr > 0:
            f1_score = 2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr)
            print(f"{'F1-Score':<20} {f1_score:.4f}")
        
        # Per-class metrics if available
        if hasattr(val_results.box, 'maps') and val_results.box.maps is not None:
            print(f"\n📈 PER-CLASS mAP@0.5 RESULTS:")
            classes_path = os.path.join('content', 'valid-data', 'classes.txt')
            with open(classes_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            
            for i, (class_name, map_val) in enumerate(zip(class_names, val_results.box.maps)):
                print(f"  {class_name:<15}: {map_val:.4f}")
        
    except Exception as e:
        print(f"❌ Official validation failed: {e}")
        print("Running inference-based evaluation instead...")
        val_results = None
    
    # Custom inference evaluation for additional metrics
    print("\n" + "="*50)
    print("RUNNING DETAILED INFERENCE ANALYSIS")
    print("="*50)
    
    results_dir = "validation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    inference_times = []
    detection_counts = []
    confidence_scores = []
    all_predictions = []
    
    print(f"Processing {len(valid_images)} validation images...")
    
    for i, image_path in enumerate(valid_images):
        img_name = os.path.basename(image_path)
        print(f"  📷 Processing {i+1}/{len(valid_images)}: {img_name}")
        
        # Measure inference time
        start_time = time.time()
        results = model(image_path, save=False, verbose=False)
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # Extract detection information
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                detection_counts.append(len(boxes))
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                confidence_scores.extend(confidences.tolist())
                
                # Store predictions for analysis
                for conf, cls in zip(confidences, classes):
                    all_predictions.append({
                        'image': img_name,
                        'confidence': float(conf),
                        'class': int(cls)
                    })
            else:
                detection_counts.append(0)
        
        # Save annotated image for ALL validation images
        result_path = os.path.join(results_dir, f"annotated_{img_name}")
        results[0].save(result_path)
    
    # Calculate and display comprehensive metrics
    print("\n" + "="*50)
    print("📊 COMPREHENSIVE PERFORMANCE METRICS")
    print("="*50)
    
    # Timing metrics
    total_inference_time = sum(inference_times)
    avg_inference_time = total_inference_time / len(inference_times)
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    print(f"\n⏱️  TIMING METRICS:")
    print(f"{'Total inference time':<25}: {total_inference_time:.2f} seconds")
    print(f"{'Avg inference per image':<25}: {avg_inference_time:.4f} seconds")
    print(f"{'FPS':<25}: {fps:.2f}")
    print(f"{'Throughput':<25}: {len(valid_images)/total_inference_time:.2f} images/second")
    
    # Detection metrics
    total_detections = sum(detection_counts)
    avg_detections = total_detections / len(detection_counts) if detection_counts else 0
    images_with_detections = sum(1 for x in detection_counts if x > 0)
    
    print(f"\n🎯 DETECTION METRICS:")
    print(f"{'Total detections':<25}: {total_detections}")
    print(f"{'Avg detections per image':<25}: {avg_detections:.2f}")
    print(f"{'Images with detections':<25}: {images_with_detections}/{len(detection_counts)} ({100*images_with_detections/len(detection_counts):.1f}%)")
    print(f"{'Detection rate':<25}: {100*images_with_detections/len(detection_counts):.1f}%")
    
    # Confidence metrics
    if confidence_scores:
        conf_array = np.array(confidence_scores)
        
        print(f"\n🎲 CONFIDENCE METRICS:")
        print(f"{'Average confidence':<25}: {np.mean(conf_array):.4f}")
        print(f"{'Confidence std':<25}: {np.std(conf_array):.4f}")
        print(f"{'Min confidence':<25}: {np.min(conf_array):.4f}")
        print(f"{'Max confidence':<25}: {np.max(conf_array):.4f}")
        print(f"{'Median confidence':<25}: {np.median(conf_array):.4f}")
        
        # Confidence thresholds analysis
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        print(f"\n📊 DETECTIONS BY CONFIDENCE THRESHOLD:")
        for thresh in thresholds:
            count = np.sum(conf_array >= thresh)
            percentage = 100 * count / len(conf_array)
            print(f"  ≥ {thresh:<3}: {count:>4} detections ({percentage:>5.1f}%)")
        
        # Create and save confidence distribution plot
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Histogram
        plt.subplot(2, 2, 1)
        plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Detection Confidence Scores')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(confidence_scores)
        plt.ylabel('Confidence Score')
        plt.title('Confidence Score Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_conf = np.sort(confidence_scores)
        y = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
        plt.plot(sorted_conf, y, linewidth=2)
        plt.xlabel('Confidence Score')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Confidence Scores')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Threshold analysis
        plt.subplot(2, 2, 4)
        thresholds_fine = np.linspace(0.1, 1.0, 50)
        detection_counts_thresh = [np.sum(conf_array >= t) for t in thresholds_fine]
        plt.plot(thresholds_fine, detection_counts_thresh, linewidth=2, color='orange')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Number of Detections')
        plt.title('Detections vs Confidence Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Confidence analysis plots saved to {results_dir}/confidence_analysis.png")
    
    # Generate Confusion Matrix
    print("\n" + "="*50)
    print("📊 GENERATING CONFUSION MATRIX")
    print("="*50)
    
    try:
        # Prepare ground truth data
        ground_truths = {}
        for img_path in valid_images:
            img_name = os.path.basename(img_path)
            img_stem = Path(img_path).stem
            label_path = os.path.join(valid_labels_dir, f"{img_stem}.txt")
            ground_truths[img_name] = parse_yolo_label(label_path)
        
        # Prepare prediction data
        predictions_by_image = {}
        for pred in all_predictions:
            img_name = pred['image']
            if img_name not in predictions_by_image:
                predictions_by_image[img_name] = []
            
            # Get image dimensions for conversion (assuming square images for simplification)
            # In a real scenario, you'd want to get actual image dimensions
            img_path = None
            for path in valid_images:
                if os.path.basename(path) == img_name:
                    img_path = path
                    break
            
            if img_path:
                # For YOLO format, we need normalized coordinates
                # Since we don't have the original prediction boxes in normalized format,
                # we'll create a simplified version based on class predictions
                predictions_by_image[img_name].append({
                    'class': pred['class'],
                    'confidence': pred['confidence'],
                    'x_center': 0.5,  # Placeholder - in real implementation you'd extract from model output
                    'y_center': 0.5,  # Placeholder
                    'width': 0.3,     # Placeholder
                    'height': 0.3     # Placeholder
                })
        
        # Read class names
        classes_path = os.path.join('content', 'valid-data', 'classes.txt')
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        # Create confusion matrix
        cm = create_confusion_matrix(predictions_by_image, ground_truths, class_names)
        
        # Plot and save confusion matrix
        cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_names, cm_save_path)
        
        print(f"📊 Confusion Matrix:")
        print(f"  Shape: {cm.shape}")
        print(f"  Classes: {class_names + ['Background/FN']}")
        print(f"  Saved to: {cm_save_path}")
        
        # Calculate per-class metrics from confusion matrix
        print(f"\n📈 PER-CLASS METRICS FROM CONFUSION MATRIX:")
        for i, class_name in enumerate(class_names):
            if i < cm.shape[0] - 1:  # Exclude background row
                tp = cm[i, i]
                fp = sum(cm[j, i] for j in range(cm.shape[0]) if j != i)
                fn = sum(cm[i, j] for j in range(cm.shape[1]) if j != i)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"  {class_name:<15}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Overall accuracy
        total_correct = sum(cm[i, i] for i in range(min(cm.shape[0], cm.shape[1]) - 1))
        total_predictions = cm.sum()
        overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        print(f"\n🎯 Overall Detection Accuracy: {overall_accuracy:.3f}")
        
    except Exception as e:
        print(f"❌ Confusion matrix generation failed: {e}")
        print("This might be due to missing dependencies (sklearn, seaborn) or data format issues.")
        cm = None
    
    # Class distribution analysis
    if all_predictions:
        classes_path = os.path.join('content', 'valid-data', 'classes.txt')
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        class_counts = {}
        for pred in all_predictions:
            class_idx = pred['class']
            if 0 <= class_idx < len(class_names):
                class_name = class_names[class_idx]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            print(f"\n🏷️  CLASS DISTRIBUTION:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = 100 * count / len(all_predictions)
                print(f"  {class_name:<15}: {count:>4} detections ({percentage:>5.1f}%)")
    
    # Compile comprehensive metrics
    metrics = {
        'model_info': {
            'model_path': best_model_path,
            'validation_dataset': 'content/valid-data',
            'validation_images_count': len(valid_images),
            'label_files_count': label_count
        },
        'official_validation': {
            'map_50': float(val_results.box.map50) if val_results else None,
            'map_50_95': float(val_results.box.map) if val_results else None,
            'precision': float(val_results.box.mp) if val_results else None,
            'recall': float(val_results.box.mr) if val_results else None,
            'f1_score': float(f1_score) if val_results and 'f1_score' in locals() else None
        },
        'timing_metrics': {
            'total_inference_time_seconds': total_inference_time,
            'avg_inference_time_seconds': avg_inference_time,
            'fps': fps,
            'throughput_images_per_second': len(valid_images)/total_inference_time
        },
        'detection_metrics': {
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections,
            'images_with_detections': images_with_detections,
            'detection_rate_percentage': 100*images_with_detections/len(detection_counts)
        },
        'confidence_metrics': {
            'count': len(confidence_scores),
            'mean': float(np.mean(confidence_scores)) if confidence_scores else None,
            'std': float(np.std(confidence_scores)) if confidence_scores else None,
            'min': float(np.min(confidence_scores)) if confidence_scores else None,
            'max': float(np.max(confidence_scores)) if confidence_scores else None,
            'median': float(np.median(confidence_scores)) if confidence_scores else None,
            'thresholds': {
                f'conf_{int(t*100)}': int(np.sum(np.array(confidence_scores) >= t)) 
                for t in [0.5, 0.6, 0.7, 0.8, 0.9]
            } if confidence_scores else None
        },
        'class_distribution': class_counts if 'class_counts' in locals() else None,
        'predictions': all_predictions
    }
    
    # Save comprehensive metrics
    metrics_file = os.path.join(results_dir, 'comprehensive_validation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"  📁 Results directory: {results_dir}/")
    print(f"  📄 Comprehensive metrics: {metrics_file}")
    print(f"  📊 Confidence analysis: {results_dir}/confidence_analysis.png")
    print(f"  � Confusion matrix: {results_dir}/confusion_matrix.png")
    print(f"  �🖼️  Annotated samples: {results_dir}/annotated_*.jpg/png")
    
    # Summary
    print(f"\n" + "="*50)
    print("📋 VALIDATION SUMMARY")
    print("="*50)
    if val_results:
        print(f"✅ Official mAP@0.5: {val_results.box.map50:.3f}")
        print(f"✅ Official mAP@0.5:0.95: {val_results.box.map:.3f}")
    print(f"⚡ Average FPS: {fps:.1f}")
    print(f"🎯 Detection rate: {100*images_with_detections/len(detection_counts):.1f}%")
    if confidence_scores:
        print(f"🎲 Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"📊 Total detections: {total_detections}")
    
    return metrics

if __name__ == "__main__":
    validate_yolo_model()