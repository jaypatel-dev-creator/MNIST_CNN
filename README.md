
# MNIST Digit Classification using Convolutional Neural Networks (PyTorch)

## Project Overview

This project implements a complete end-to-end **Convolutional Neural Network (CNN)** pipeline using **PyTorch** to classify handwritten digits from the **MNIST** dataset.

The notebook demonstrates a clean and professional deep learning workflow, covering dataset preparation, model design, training, validation, testing, inference, and model persistence — following real-world machine learning best practices.

## Key Features

* CNN-based multiclass classification (**10 digits: 0–9**)
* Proper **Train / Validation / Test** split
* **GPU acceleration** with CUDA support
* **Cross-Entropy loss** with raw logits
* Accuracy and loss monitoring
* Final test evaluation on unseen data
* Single-image inference
* Model checkpoint saving

## Dataset

* **Dataset:** MNIST Handwritten Digits
* **Training samples:** 60,000
* **Test samples:** 10,000
* **Image size:** 28 × 28 (grayscale)

The training data is further split into:

* **Training set:** 85%
* **Validation set:** 15%

## Model Architecture

The CNN architecture consists of two main components:

### Convolutional Feature Extractor

* Convolutional (`Conv2D`) layers with **ReLU** activations
* **MaxPooling** layers for spatial downsampling

### Classifier Head

* Fully connected (dense) layers
* Output layer with **10 logits** (one per digit class)

No **Softmax** is applied inside the model. `CrossEntropyLoss` internally handles the softmax operation during training.

## Training Strategy

* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Loss Function:** CrossEntropyLoss
* **Batch Size:** 64
* **Epochs:** 5

### Metrics Tracked

* Training loss and accuracy
* Validation loss and accuracy
* Final test loss and accuracy

## Workflow Breakdown

The notebook follows a structured machine learning pipeline:

1. Device configuration (CPU / GPU)
2. Dataset loading and preprocessing
3. Train–validation split
4. CNN model definition
5. Training loop
6. Validation loop
7. Final test evaluation
8. Inference on a sample image
9. Model saving

## Results

The trained CNN achieves strong performance on the MNIST test set, demonstrating the effectiveness of convolutional architectures for image classification tasks.

Exact metrics can be found in the notebook output cells.

## Purpose

This project is designed to:

* Demonstrate correct usage of **Convolutional Neural Networks (CNNs)** in PyTorch
* Follow **industry-standard machine learning workflows**
* Serve as a **portfolio-ready deep learning project**
* Be suitable for **technical interviews and code reviews**

## Technologies Used

* Python
* PyTorch
* Torchvision
* Google Colab / Jupyter Notebook
