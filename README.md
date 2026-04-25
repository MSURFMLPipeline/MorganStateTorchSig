# Introduction
This Github project is showcasing a RFML dataset generation framework through a collaboration between Morgan State senior Electrical and Computer Engineering students at Morgan State University and the SME (Subject Matter Expert) advisor at the Aerospace Coporation. The student team includes Corey Hicks, Zimiego Smith, and Jamari Richardson working under the guidance of a subject matter expert from the Aerospace Coporation, Donna Branchevsky and an Morgan State faculty advisor, Dr. Willie Thompson. 

![alt text](image.png) Team member #1: Zimiego Smith                                                                  ![alt text](image-1.png) Team member #2: Jamari Richardson                                                            ![alt text](image-2.png) Team member #3: Corey Hicks

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
git clone https://github.com/MSURFMLPipeline/MorganStateTorchSig.git
cd MorganStateTorchSig
pip install -e .
```
2. Install the following dependencies
```
pip install -r requirements.txt
```
3. Configure yaml files (Edit as needed)
```
config.yaml
yolo_Detector.yaml
yolo_Detector_2.yaml
yolo_Detector_3.yaml
```
| Files | Purpose |
|-------|---------|
|`config.yaml`| Has the dataset settings for dataset generation, training/testing/validation parameters, class list, and SNR|
|`yolo_Detector.yaml`| This is the YOLO Configuration File for Clean Signals|
|`yolo_Detector_2.yaml`| This is the YOLO Configuration File for EMI Signals |
|`yolo_Detector_3.yaml`| This is the YOLO Configuration File for Clean+EMI Signals |

# Usage

Users are able to use this pipeline to generate different data. They are able to train a YOLOv8n model on different data types
Within the `config.yaml` file, a user will be able to select one of the three data options to generate data (Clean, EMI, or EMI+Clean)

1. Selecting Data Option Mode for Synthethic Data Generation
- A user must select one of the data options in the `config.yaml` file

| Dataset Modes | Description |
|-------|---------|
|`Clean`| Generates clean signals only |
|`EMI`|Generates EMI (tone) interference with awgn signals only| 
|`Clean+EMI`| Generates mixed clean + EMI signals. Where EMI is a tone with awgn and wireless imapairments| 

2. Run the pipeline
- python `main.py`

# Core Scripts

- `MSUTorch/model/evaluation.py`: Has evaluation metrics for the train/test/validation datasets
- `MSUTorch/data_generation/clean.py`: Allows users to generate clean data from the `data.py` file 
- `MSUTorch/data_generation/clean_plus_emi.py`: Allows users to generate clean+emi data from the `data.py` file
- `MSUTorch/data_generation/emi.py`: Allows users to generate emi data from `data.py` file
- `MSUTorch/models/yolo_model.py`: Loads in YOLOv8n pretrained model
- `MSUTorch/dataset_generation/data.py`: Generates data and writes to disk
- `MSUTorch/dataset_generation/transforms.py`: Reads from disk and handels YOLO Label and Spectrogram transforms
- `MSUTorch/models/evaluation.py`: Handels evaluation of the training/testing/validation datasets in the YOLOv8n model

# Future Works

1. Working on allowing users to build their own EMI in complaince to FCC standards
2. Making an option for users to train their own model
3. Greater model diversity and EMI diversity
4. Testing pipeline for over-the-air transmission 

# License
MorganStateTorchSig is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. MorganStateTorchSig has no connection to MIT, other than through the use of this license.

# Citing Torchsig
"""
bibtext
@misc{torchsig,
  title={Large Scale Radio Frequency Signal Classification},
  author={Luke Boegner and Manbir Gulati and Garrett Vanhoy and Phillip Vallance and Bradley Comar and Silvija Kokalj-Filipovic and Craig Lennon and Robert D. Miller},
  year={2022},
  archivePrefix={arXiv},
  eprint={2207.09918},
  primaryClass={cs-LG},
  note={arXiv:2207.09918}
  url={https://arxiv.org/abs/2207.09918}
}
"""
