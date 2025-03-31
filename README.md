# Face Recognition System
---
A real-time face recognition system using deep learning and machine learning. This project captures images, processes them into embeddings, trains an SVM classifier, and performs real-time face recognition via a webcam.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## **Features**

✅ Capture and store face images for training  
✅ Preprocess images and extract face embeddings using FaceNet  
✅ Augment images to improve model performance  
✅ Train an SVM classifier for face identification  
✅ Real-time face recognition with bounding boxes and confidence scores  

## **Installation**

### **Prerequisites**

Ensure you have the following installed:
- Python 3.8+
- OpenCV
- NumPy
- Scikit-learn
- Joblib
- Keras & TensorFlow
- FaceNet (keras_facenet)

Install dependencies using:

```bash
pip install -r requirements.txt
```

## **Usage**

### **Step 1: Capture Photos**

Run the script to capture face images:

```bash
python capture_photos.py
```

Press ‘c’ to capture an image and ‘q’ to quit.

### **Step 2: Preprocess and Extract Embeddings**

```bash
python data_preprocessing.py
```

This script detects faces, applies augmentations, and generates embeddings.

### **Step 3: Train the Model**

```bash
python train_model.py
```

This trains an SVM classifier using the extracted embeddings and saves the model.

### **Step 4: Run Face Recognition**

```bash
python face_recognition.py
```

The script will open a webcam feed and recognize faces in real time.
