# MNIST Digit Classification using Convolutional Neural Networks (PyTorch)

## Project Overview

This project implements a complete end-to-end **Convolutional Neural Network (CNN)** pipeline using **PyTorch** to classify handwritten digits from the **MNIST** dataset.

The notebook demonstrates a clean, reproducible, and checkpoint-enabled deep learning workflow, covering dataset preparation, model design, training, validation, testing, inference, and model persistence — following real-world machine learning engineering best practices.


## Run Notebook in Google Colab

Click below to run the notebook instantly without any local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/jaypatel-dev-creator/MNIST_CNN/blob/main/MNISTCNN.ipynb
)

## Key Features

* CNN-based multiclass classification (**10 digits: 0–9**)
* Proper **Train / Validation / Test** split for unbiased evaluation
* **Full reproducibility controls** using fixed random seeds (PyTorch, NumPy, CUDA)
* **GPU acceleration** with CUDA support
* **Cross-Entropy loss** with raw logits
* Training and validation **loss and accuracy monitoring**
* **Best model checkpoint saving** based on validation accuracy
* **Best model reload** before final test evaluation
* Final test evaluation on unseen data
* Separate **inference pipeline** for real-world predictions
* Model persistence for reproducibility and deployment readiness

## Dataset

* **Dataset:** MNIST Handwritten Digits
* **Training samples:** 60,000
* **Test samples:** 10,000
* **Image size:** 28 × 28 (grayscale)

The training data is further split into:

* **Training set:** 85%
* **Validation set:** 15%

The split is performed using a fixed random seed to ensure reproducibility.

## Model Architecture

The CNN architecture consists of two main components:

### Convolutional Feature Extractor

* Convolutional (`Conv2D`) layers with **ReLU** activations
* **MaxPooling** layers for spatial downsampling
* Hierarchical feature extraction from input images

### Classifier Head

* Fully connected (dense) layers
* Output layer with **10 logits** (one per digit class)


## Training Strategy

* **Optimizer:** Adam
* **Learning Rate:** 0.001
* **Loss Function:** CrossEntropyLoss
* **Batch Size:** 64
* **Epochs:** 5
* **Device:** CPU / GPU (CUDA)

### Metrics Tracked

* Training loss and accuracy
* Validation loss and accuracy
* Best validation accuracy for checkpoint saving
* Final test loss and accuracy

## Workflow Breakdown

The notebook follows a structured and reproducible deep learning pipeline:

1. Reproducibility setup (random seeds for PyTorch, NumPy, CUDA)
2. Device configuration (CPU / GPU)
3. Dataset loading and preprocessing with normalization
4. Train–validation split with fixed seed
5. CNN model definition
6. Training loop with metric tracking
7. Validation loop with checkpoint saving
8. Best model reload
9. Final test evaluation
10. Inference pipeline for prediction on individual images
11. Model saving for reuse and deployment

## Results

The trained CNN achieves strong performance on the MNIST dataset:

* **Best Validation Accuracy:** 98.84%
* **Final Training Accuracy:** 99.44%
* **Final Validation Accuracy:** 98.72%
* **Test Accuracy:** ~98%+

The model demonstrates fast convergence, strong generalization, and stable training behavior.

## Purpose

This project is designed to:

* Demonstrate correct implementation of **Convolutional Neural Networks (CNNs)** in PyTorch
* Follow **industry-standard deep learning training and validation workflows**
* Demonstrate **model checkpointing and reproducibility practices**
* Showcase a complete **training, evaluation, and inference pipeline**


## Technologies Used


* PyTorch
* NumPy
* Google Colab
