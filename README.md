# Federated Learning for Autonomous Driving

## Project Overview
This repository contains the implementation and analysis of key aspects of **Federated Learning (FL)** applied to autonomous driving. The project focuses on developing privacy-preserving models for edge devices like autonomous vehicles while addressing challenges such as domain shifts, statistical heterogeneity, and unlabeled client data. This project was completed by Valerio Mastrianni, Lal Akin and Riccardo Zanchetta.

The repository includes code implementations for:
1. **Step 1: Centralized Baseline**
2. **Step 2: Supervised Federated Learning**
3. **Step 5: YOLOv8 Ensemble Learning**

Conceptual overviews are provided for:
- **Step 3: FFreeDA Pre-Training Phase**
- **Step 4: Federated Self-Training using Pseudo-Labels**

---

## Problem Breakdown

### **Step 1: Centralized Baseline**
- A benchmark model trained on the IDDA dataset for comparison with federated settings.
- Explored data augmentation techniques and their impact on model performance.
- Key insights:
  - Random cropping was the most effective augmentation method.
  - Best mIoU achieved: **0.643** on the same-domain test set.

### **Step 2: Supervised Federated Learning**
- Decentralized training on edge devices using the FedAvg algorithm.
- Key experiments:
  - Tested different numbers of clients per communication round.
  - Evaluated the effect of learning rate schedulers (LRS1 and LRS2).
- Results:
  - Optimal performance with 8 clients per round and a higher number of local epochs.
  - Best mIoU achieved: **0.622** on the same-domain test set.

### **Step 5: YOLOv8 Ensemble Learning**
- Combined **DeepLabV3** (for semantic segmentation) and **YOLOv8** (for object detection) models.
- Improved segmentation performance, particularly for critical classes (e.g., pedestrians, vehicles).
- Results:
  - Significant gains in Mean IoU for Cityscapes Dataset:
    - **Person IoU**: +36%
    - **Motorcycle IoU**: +28%
    - **Bicycle IoU**: +38.8%

---

### **Steps 3 and 4 (Conceptual Overview)**
#### **Step 3: FFreeDA Pre-Training Phase**
- Implemented Fourier Domain Adaptation (FDA) to align domain-specific data distributions.
- Leveraged labeled GTA5 dataset for pre-training and tested style-transferred data.

#### **Step 4: Federated Self-Training**
- Used pseudo-labeling to address the challenge of unlabeled client data.
- Teacher-student model setup to generate pseudo-labels and refine predictions iteratively.

---

## Tools and Techniques
- **Python**:
  - `PyTorch`, `NumPy`, `Matplotlib`: For model training, data handling, and visualization.
- **Deep Learning Frameworks**:
  - **DeepLabV3**: For semantic segmentation.
  - **YOLOv8**: For object detection in the ensemble learning setup.
- **Optimization**:
  - FedAvg algorithm for federated learning.
  - FDA for domain adaptation.

---

## Results Summary
- **Centralized Baseline**:
  - Achieved best mIoU: **0.643** with random cropping.
- **Supervised Federated Learning**:
  - Best performance: **0.622 mIoU** with 8 clients per round.
- **YOLOv8 Ensemble Learning**:
  - Enhanced segmentation and object detection, especially for critical classes.
