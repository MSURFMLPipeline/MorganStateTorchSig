# Introduction
This Github project is showcasing a RFML dataset generation framework through a collaboration between Morgan State senior Electrical and Computer Engineering students at Morgan State University and the SME (Subject Matter Expert) advisor at the Aerospace Coporation. The student team includes Corey Hicks, Zimiego Smith, and Jamari Richardson working under the guidance of a subject matter expert from the Aerospace Coporation, Donna Branchevsky and an Morgan State faculty advisor, Dr. Willie Thompson. 

![alt text](image.png) Team member #1: Zimiego Smith

![alt text](image-1.png) Team member #2: Jamari Richardson

![alt text](image-2.png) Team member #3: Corey Hicks

# Prerequisites
For this project we recommend using the following: 

- For windows users WSL Ubuntu (Windows Subsystem for Linux) with 4 or more CPU cores
- Python ≥ 3.10
- Ubuntu ≥ 22.04
- Hard drive storage with 1 TB
- GPU with ≥ 16 GB storage

# Getting started with MSUTorch-RF Signal Detection Pipeline

The pipeline is set with the `MSUTorch/` directory and the `data_generation/` and `models/` folders 

The `examples/` directory is to help those who are new to object oriented programming gain experience and develop their understanding. This gave us the background we needed to develop the pipeline

# Setup

1. Clone the Repository
```
git clone https://github.com/MorganStateTorchSig/MSUTorch.git
cd MSUTorch
pip install -e .
```
2. Install the following dependencies
```
pip install -r requirements.txt
```
3. Configure yaml files (Copy as needed)
```
cp config.yaml
cp yolo_Detector.yaml
cp yolo_Detector_2.yaml
cp yolo_Detector_3.yaml
```
| Files | Purpose |
|-------|---------|
|`config.yaml`| Has the dataset settings for dataset generation, training/testing/validation parameters, class list, and SNR|
|`yolo_Detector.yaml`| This is the YOLO Configuration File for Clean Signals|
|`yolo_Detector_2.yaml`| This is the YOLO Configuration File for EMI Signals |
|`yolo_Detector_3.yaml`| This is the YOLO Configuration File for Clean+EMI Signals |

# Usage

Within the `config.yaml` file, a user will be able to select one of the three data options to generate data (Clean, EMI, or EMI+Clean)

1. Selecting Data Option for Synthethic Data Generation
- A user must select one of the data options in the `config.yaml` file
2. Run the pipeline
- python `main.py`



