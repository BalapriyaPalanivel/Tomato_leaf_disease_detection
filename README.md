# Tomato Leaf Disease Detection

This project is designed to detect and classify 10 different diseases in tomato leaves using TensorFlow and OpenCV (cv2). The model leverages deep learning techniques to accurately identify the disease present in the leaf, helping farmers and gardeners in early diagnosis and treatment.

## Project Overview

Tomato plants are susceptible to various diseases that can severely impact crop yield and quality. This project aims to automate the detection of these diseases using computer vision and deep learning, providing a reliable and efficient tool for farmers.

The model can identify the following 10 diseases:
1. Bacterial spot
2. Early blight
3. Late blight
4. Leaf Mold
5. Septoria leaf spot
6. Spider mites
7. Target spot
8. Yellow Leaf Curl Virus
9. Mosaic Virus
10. Healthy

## Dataset

The dataset used for training and testing the model consists of labeled images of tomato leaves, each annotated with one of the 10 possible diseases or marked as healthy. The dataset is split into training, validation, and test sets to ensure the model generalizes well to unseen data.

## Model Architecture

The model is built using TensorFlow and leverages a convolutional neural network (CNN) architecture. The key features include:
- Multiple convolutional layers for feature extraction
- Max-pooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax activation for multi-class classification

OpenCV (cv2) is used for image preprocessing, including resizing, normalization, and augmentation.

## Usage

1. **Training the Model:**
   - Place the dataset in the appropriate directory (`/data`) and ensure it is correctly structured.
   - Run the training script:
     ```bash
     python train.py
     ```

2. **Predicting on New Images:**
   - Use the trained model to predict the disease on new tomato leaf images:
     ```bash
     python predict.py --image_path path/to/leaf_image.jpg
     ```

3. **Evaluating the Model:**
   - To evaluate the model on the test set, run:
     ```bash
     python evaluate.py
     ```
