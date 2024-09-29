# Face-Recognition

This project implements face detection and recognition using OpenCV's Haar Cascade Classifier and custom face recognition models. The system captures and trains faces, then recognizes them in real-time using `datacollect_and_training.py` for data collection and training, and `testmodel.py` for testing the recognition model. Additionally, AWS S3 is used to upload and fetch images for the model.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Files and Directories](#files-and-directories)
- [Usage](#usage)
- [AWS S3 Integration](#aws-s3-integration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project utilizes OpenCV’s `haarcascade_frontalface_default.xml` for face detection and a custom model for face recognition. It includes scripts for data collection, model training, and testing. AWS S3 is used to store and fetch images for training and recognition purposes.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- AWS SDK for Python (boto3)
- scikit-learn
- TensorFlow (if using neural networks for face recognition)

Install the required libraries using:

```bash
pip install opencv-python numpy boto3 scikit-learn tensorflow


