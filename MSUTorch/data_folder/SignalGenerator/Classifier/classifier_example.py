"""Classifier Example
Utilizes Pytorch which is an open source library and implifies deep learning and machine learning

Tensors are used as a multi-dimensional array and is fundamental for structuring data in ML and deep learning. They are used in PyTorch to prepare data for the deep learning model. 

Batching is a data processing technique where data is divided into smaller groups called batches. Batch_Size is defining how many data samples are in each batch.


"""


# Variables
"""Our Iniitial Analysis: Defining All the neccessary variables and creating directory"""
from torchsig.signals.signal_lists import TorchSigSignalLists #Config file that has dictionaries of signal classes 
from torchsig.transforms.transforms import ComplexTo2D # Converts IQ data into two channels,real and imaginary parts
import os # interacting with operating system

from torch import Tensor 

root = "./datasets/classifier_example" #Directory path
os.makedirs(root, exist_ok=True)  #Creating directory from root and if it exist nothing happens
os.makedirs(root + "/train", exist_ok=True) #Subdirectory for training
os.makedirs(root + "/val", exist_ok=True) #Subdirectory for validation
os.makedirs(root + "/test", exist_ok=True) #Subdirectory for testing

fft_size = 256
num_iq_samples_dataset = fft_size ** 2 #fft_size^2
class_list = TorchSigSignalLists.all_signals #All the signals from the TorchsigSignalsList
family_list = TorchSigSignalLists.family_list #List the values from family_dict(The family_dict is a dictionary that has all signal types and families) from TorchSignalList
num_classes = len(class_list) #Length of the all the signals from TorchsigSignalsList 
num_samples_train = len(class_list) * 10 # roughly 10 samples per class 
num_samples_val = len(class_list) * 2 # roughly 2 samples per class
impairment_level = 0
seed = 123456789
 # IQ-based mod-rec only operates on 1 signal
num_signals_max = 1
num_signals_min = 1

# ComplexTo2D turns a IQ array of complex values into a 2D array, with one channel for the real component, while the other is for the imaginary component
transforms = [ComplexTo2D()]

"""-------------------------------------------------------------"""
#Create the Dataset
"""Initial Analysis: Has the libraries from the create dataset example, Just creating DatasetMetadata but now it is splitting the Dataset into training and validation 
Additionally there is now a dataloader for train and validation

Classes: 1. DatasetMetadata, 2. WorkerSeedingDataLoader, 3. TorchSigIterableDataset, 4. DatasetCreator 

1. DatasetMetadata is used for defining dataset parameter values
2. WorkerSeedingDataLoader is used for testing, debugging and reproducing experiments. DataLoader seeds(the seeds are random unique) each worker process differently using a shared seed. From Seedable
- This loader prohibits external worker_init_fn definitions and sets its own init function to ensure reproducible randomness in multi-worker pipelines.
- Workers are responsible for doing task at the same time
- Seed controls randomness 
3.TorchSigIterableDataset is used for generating synthetic data infinitely in memory using randomized DatasetMetadata values
4.DatasetCreator is used to create a dataset which is then saved to disk in batches
5.

"""
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.writer import DatasetCreator #

dataset_metadata = DatasetMetadata(
    num_iq_samples_dataset = num_iq_samples_dataset,
    fft_size = fft_size,
    class_list = class_list,
    num_signals_max = num_signals_max,
    num_signals_min = num_signals_min,
)

train_dataset = TorchSigIterableDataset(dataset_metadata, transforms=transforms, target_labels=None)
val_dataset = TorchSigIterableDataset(dataset_metadata, transforms=transforms, target_labels=None)

train_dataloader = WorkerSeedingDataLoader(train_dataset, batch_size=4, collate_fn = lambda x: x) #Batch size is how many samples you want loaded at one time #Collate is collecting and combining the samples into batches
val_dataloader = WorkerSeedingDataLoader(val_dataset, collate_fn = lambda x: x) 

#print(f"Data shape: {data.shape}")
#print(f"Targets: {targets}")
# next(train_dataset)

dc = DatasetCreator(
    dataloader=train_dataloader,
    root = f"{root}/train",
    overwrite=True,
    dataset_length=num_samples_train
)
dc.create()


dc = DatasetCreator(
    dataloader=val_dataloader,
    root = f"{root}/val",
    overwrite=True,
    dataset_length=num_samples_val
)
dc.create()
train_dataset = StaticTorchSigDataset(
    root = f"{root}/train",
    target_labels=["class_index"]
)
val_dataset = StaticTorchSigDataset(
    root = f"{root}/val",
    target_labels=["class_index"]
)

train_dataloader = WorkerSeedingDataLoader(train_dataset, batch_size=4)
val_dataloader = WorkerSeedingDataLoader(val_dataset)

print(train_dataset[0])
next(iter(train_dataloader))
"""-----------------------------------------------------------------------------"""
#Create the Model


from torchsig.models import XCiTClassifier
from torchinfo import summary

model = XCiTClassifier(
    input_channels=2,
    num_classes=num_classes,
)
summary(model)

"""----------------------------------------------------"""
#Train the Model


import torch
import pytorch_lightning as pl

num_epochs = 1

trainer = pl.Trainer(
    limit_train_batches=10,
    limit_val_batches=5,
    max_epochs = num_epochs,
    accelerator =  'gpu' if torch.cuda.is_available() else 'cpu',
    devices = 1
)

trainer.fit(model, train_dataloader)

"""---------------------------------------------------------------------------------"""
#Test the Model
"""


"""
from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.utils.writer import DatasetCreator, default_collate_fn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
torch.cuda.empty_cache()

test_dataset_size = 10


dataset_metadata_test = DatasetMetadata(
    num_iq_samples_dataset = num_iq_samples_dataset,
    fft_size = fft_size,
    class_list = class_list,
    num_samples=test_dataset_size,
    seed = 123456788, # different than train
    num_signals_max = num_signals_max,
    num_signals_min = num_signals_min
)
# print(dataset_metadata_test)
dataset = TorchSigIterableDataset(dataset_metadata_test, transforms=transforms, target_labels=None,)#["class_index"])

dataloader = WorkerSeedingDataLoader(dataset, num_workers=1, batch_size=1, collate_fn = lambda x: x)#default_collate_fn)

dc = DatasetCreator(
    dataloader=dataloader,
    root = f"{root}/test",
    overwrite=True,
    dataset_length=100
)
dc.create()

test_dataset = StaticTorchSigDataset(
    root = f"{root}/test",
    target_labels=["class_index"]
)

data, class_index = test_dataset[0]
print(f"Data shape: {data.shape}")
print(f"Targets: {class_index}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, class_index = test_dataset[0]
# move to model to the same device as the data
model.to(device)
# turn the model into evaluation mode
model.eval()
with torch.no_grad(): # do not update model weights
    # convert to tensor and add a batch dimension
    data = torch.from_numpy(data).to(device).unsqueeze(dim=0)
    # have model predict data
    # returns a probability the data is each signal class
    pred = model(data)
    # print(pred) # if you want to see the list of probabilities

    # choose the class with highest confidence
    predicted_class = torch.argmax(pred).cpu().numpy()
    print(f"Predicted = {predicted_class} ({class_list[predicted_class]})")
    print(f"Actual = {class_index} ({class_list[class_index]})")


# We can do this over the whole test dataset to check to accurarcy of our model
predictions = []
true_classes = []
num_correct = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for sample in test_dataset:
    data, actual_class = sample
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = torch.from_numpy(data).to(device).unsqueeze(dim=0)
        pred = model(data)
        predicted_class = torch.argmax(pred).cpu().numpy()
        predictions.append(predicted_class)
        true_classes.append(actual_class)
        if predicted_class == actual_class:
            num_correct += 1

# try increasing num_epochs or train dataset size to increase accuracy
print(f"Correct Predictions = {num_correct}")
print(f"Percent Correct = {num_correct / len(test_dataset)}%")

# We can also plot a confusion matrix using Sklearn's confusion matrix tool
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matrix = confusion_matrix(true_classes, predictions, labels=list(range(len(family_list))))
disp = ConfusionMatrixDisplay(matrix, display_labels=family_list)
disp.plot()
