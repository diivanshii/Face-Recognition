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

Setup Instructions
1. Clone the repository:
bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Download Haar Cascade Classifier:
The pre-trained face detection model is based on Haar Cascades. You can download the haarcascade_frontalface_default.xml from OpenCV or use the one provided in the repository.

3. Configure AWS S3:
Set up your AWS S3 credentials. Ensure that you have an S3 bucket ready for storing the images.

Create or modify the aws_config.json file in the root directory:

json
{
  "bucket_name": "your-bucket-name",
  "access_key": "your-access-key",
  "secret_key": "your-secret-key",
  "region": "your-region"
}
4. Run data collection and model training:
Use the datacollect_and_training.py script to collect images, store them in S3, and train the recognition model.

bash
python datacollect_and_training.py
5. Test the model:
Run the testmodel.py script to test the face recognition in real-time using the webcam or pre-uploaded images from S3.

bash
python testmodel.py
Files and Directories
haarcascade_frontalface_default.xml: Pre-trained model for face detection using Haar cascades.
datacollect_and_training.py: Script to collect face data, upload to AWS S3, and train the recognition model.
testmodel.py: Script to recognize faces in real-time using the trained model.
aws_config.json: Configuration file for AWS S3 credentials.
Usage
1. Data Collection and Model Training:
Run the following command to start collecting face data and train the model:

bash
python datacollect_and_training.py
The script will:

Capture face images from the webcam.
Upload images to AWS S3 for backup.
Train the face recognition model using the collected data.
2. Testing the Recognition Model:
Run the following command to test the trained model on new faces:

bash
python testmodel.py
This will:

Load the model.
Detect and recognize faces in real-time using the webcam or fetched images from AWS S3.
AWS S3 Integration
Uploading Images to S3:
Images captured during the data collection phase are automatically uploaded to the specified S3 bucket using boto3. The S3 bucket name and credentials are configured in the aws_config.json file.

Fetching Images from S3:
When testing the model, the script can fetch images from S3 if specified, allowing for remote access to test data stored in the cloud.

Setting up AWS S3 with boto3:
Install boto3 using pip:

bash
pip install boto3
Configure your AWS credentials using the aws_config.json file as mentioned above.

Contributing
Feel free to contribute to this project by submitting a pull request. Ensure that your code follows the PEP8 style guide and includes appropriate tests.

License
This project is licensed under the MIT License.
