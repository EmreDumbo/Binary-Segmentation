# 🚗 Binary Car Segmentation using Carvana Dataset

## 📌 Project Overview
This project focuses on **binary segmentation** of cars using the **Carvana Image Masking Dataset**. The goal is to train a deep learning model to accurately distinguish between **cars** and the **background** in images. The project utilizes **U-Net** for segmentation and achieves high accuracy in car detection.

## 📂 Dataset
We used the **Carvana Image Masking Dataset**, which consists of high-resolution car images along with their corresponding binary masks:
- **Input:** Car images
- **Output:** Binary masks (car vs. background)

## 🏗 Model Architecture
We implemented **U-Net**, a popular convolutional neural network (CNN) architecture for segmentation tasks. The architecture consists of:
- **Encoder:** Feature extraction using convolutional layers
- **Bottleneck:** Learning compact representations
- **Decoder:** Upsampling and refinement to generate segmentation masks

## 🛠 Implementation Steps
1. **Data Preprocessing:**
   - Resized images to a suitable resolution
   - Normalized pixel values
   - Applied augmentations (flip, rotate, etc.)
2. **Model Training:**
   - Used **U-Net** for segmentation
   - Optimized with **Dice loss** and **Binary Cross-Entropy (BCE) loss**
   - Trained using **Adam optimizer**
3. **Evaluation:**
   - Measured **IoU (Intersection over Union)** and **Dice coefficient**
   - Visualized predictions vs. ground truth masks
4. **Inference:**
   - Performed segmentation on test images
   - Evaluated real-world performance

## 🚀 Results
The model successfully segments cars with **high accuracy**, achieving:
- **Val Accuracy:** ~0.95
- **Val Loss:** ~0.94


## 📦 Dependencies
To run this project, install the required dependencies:
```bash
pip install torch torchvision numpy opencv-python matplotlib albumentations
```

## 📜 Usage
### **1️⃣ Train the Model**
```bash
python train.py
```
### **2️⃣ Evaluate the Model**
```bash
python evaluate.py
```


## 📌 Future Improvements
- Experimenting with **multi-class segmentation** (e.g., different car parts)
- Trying **other architectures** like DeepLabV3
- Deploying the model as a **web application**

## 🤝 Contributors
- **[Emre Dumbo]**

## ⭐ Acknowledgments
- **Carvana Image Masking Dataset**
- **U-Net paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"**
