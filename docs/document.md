# Technical Document for SEPCO Look of Success (LOS) Project

## Metadata
**Prepared by**: [Your Name]  
**Last Updated**: September 22, 2025  
**Version**: 0.0.1  
**Version Description**: Initial version of the SEPCO LOS project documentation.  

---

## 1. Overview

### Problem Statement
The SEPCO Look of Success (LOS) project aims to address challenges in automating the detection of storefront compliance with predefined guidelines. Manual processes are inefficient and prone to errors, especially when dealing with large datasets. This project provides an automated solution for verifying if specific marketing materials are present and if they are in the correct locations.

### Solution
The SEPCO LOS project is a Python-based application that leverages the YOLOv9c model for training and inference. It supports batch processing, real-time feedback, and efficient data handling. The system ensures accuracy, scalability, and ease of use for storefront compliance verification tasks.

---

## 2. System Architecture and Workflow

### Architecture
- **Backend**: Python-based processing engine with YOLOv9c for training and inference.
- **Data Storage**: Local storage for input data, training results, and logs.

### Workflow
1. **Initialization**:
   - The system begins by loading the YOLOv9c model and configuration settings from `data.yaml`. These settings include parameters for training, validation, and inference.

2. **Data Preparation**:
   - The system automatically splits the dataset into training and validation sets based on a predefined ratio (80% training, 20% validation).
   - Paths to training and validation images are saved in `train.txt` and `val.txt` respectively.

3. **Training**:
   - The YOLOv9c model is trained using the prepared dataset. Training parameters include:
     - **Epochs**: 100
     - **Image Size**: 320
     - **Batch Size**: 4
     - **Device**: CPU
   - Training results, including metrics and weights, are saved in the `runs/detect` directory.

4. **Inference**:
   - The system uses the best-trained model (`best.pt`) to run inference on test images located in the `test_data` directory.
   - Inference times are measured and reported for each image.

5. **Output**:
   - Training and inference results are saved in the respective directories. Logs are archived for auditing and debugging purposes.

---

## 3. Data Input and Output

### Input
- **Image Files**: Supported formats include PNG, JPG, JPEG, BMP, and TIFF.
- **Configuration**: Settings are loaded from `data.yaml`. Key configurable parameters include:
  - **`train`**: Path to the training dataset.
  - **`val`**: Path to the validation dataset.
  - **`nc`**: Number of classes.
  - **`names`**: List of class names.

### Output
- **Training Results**:
  - Metrics and weights are saved in the `runs/detect` directory.
- **Inference Results**:
  - Inference times and outputs are reported for test images.
- **Logs**: Processing logs are saved for debugging and auditing purposes.

---

## 4. Confidentiality Notice

This document and the SEPCO LOS project contain proprietary information and are intended solely for authorized personnel. Unauthorized access, distribution, or modification of this document or the software is strictly prohibited. All data processed by the system is handled with strict confidentiality, and temporary files are securely deleted after use. For further inquiries, contact the project administrator.

--- 

**End of Document**