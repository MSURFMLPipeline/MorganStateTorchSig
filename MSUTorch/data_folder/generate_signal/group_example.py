# QPSK Clean Data 
import numpy as np
import matplotlib.pyplot as plt
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.transforms.transforms import Spectrogram
from torchsig.utils.writer import DatasetCreator, default_collate_fn
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from tqdm.notebook import tqdm
from torchsig.signals.signal_lists import TorchSigSignalLists
from torchsig.datasets.datasets import StaticTorchSigDataset


"""SNR is the same for the max and min and is set at 30 to get the ideal clean QPSK signal"""
#Metadata
num_iq_samples_dataset = 8192
fft_size = 256
num_signals_min = 1
num_signals_max = 1
snr_db_min = 30  
snr_db_max = 30
class_list = ["qpsk"]

#------------------
# Dataset Metadata
#------------------
dataset_metadata = DatasetMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    fft_size=fft_size,
    num_signals_min=num_signals_min,
    num_signals_max=num_signals_max,
    snr_db_min=snr_db_min,
    snr_db_max=snr_db_max,
    class_list=class_list,
)


# Creating Dataset 
#------------------------
"""This block of code prints modulation, class list, and SNR"""
# Without target_labels, returns Signal objects with rich metadata
#------------------------
dataset = TorchSigIterableDataset(dataset_metadata = dataset_metadata)  # Generates signals using randomized parameters values from dataset_metadata values

for i in range(5): #Uses a for loop from 1 to 5 to print out the number of max and min signals
    signal = next(dataset) #Python Iterator #Allows you to go through a sequence of signals until you reach the end #creates one signal object composed of component signals
    
    print(f"IQ Data shape: {signal.data.shape}") # Shows the dimensions of the array for the IQ data #.data is used to get the raw data # Gets the complex IQ samples that are in the form of an array

    print(f"Component Signals: {len(signal.component_signals)}") #Len shows number of items in an object 

#Component_signal has the individual components of the full signal, e.g. smaller individual signals collected together in a wideband signal. Defaults to []. Such as phase, frequency, amplitude, etc
    
    # Access metadata from component signals
    for j, comp_signal in enumerate(signal.component_signals):
        print(f"  Signal {j}: {comp_signal.metadata.class_name}, SNR: {comp_signal.metadata.snr_db:.1f}dB")
    print()

#-------------------
# Time domain plot 
#-------------------
dataset_time_series_metadata = DatasetMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    fft_size=fft_size,
    num_signals_min=num_signals_min,
    num_signals_max=num_signals_max,
    snr_db_min=snr_db_min,
    snr_db_max=snr_db_max,
    class_list=class_list,
)



dataset_time_series = TorchSigIterableDataset(
dataset_metadata = dataset_time_series_metadata, 
target_labels=["class_index"]
)

data, metadata = next(dataset_time_series) # Splitting the dataset time series into data and metadata, where data would be the real and imaginary parts. Metadata would store the parameter values
t = np.arange(0,len(data))/dataset_time_series_metadata.sample_rate # np.arange is going from 0 to the length of the data array minus 1 divided by the sample rate

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(1,1,1)
ax.plot(t,np.real(data),alpha=0.5,label='Real')
ax.plot(t,np.imag(data),alpha=0.5,label='Imag')
ax.set_xlim([t[0], t[-1]])
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Amplitude')
ax.set_title('Time Domain')
ax.grid()
plt.show()



#-------------------------------
# Spectrogram Plot (Magnitude)
#-------------------------------
dataset_finite_spectrogram_metadata = DatasetMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    fft_size=fft_size,
    num_signals_min=num_signals_min,
    num_signals_max=num_signals_max,
    snr_db_min=snr_db_min,
    snr_db_max=snr_db_max,
    class_list=class_list,
)

dataset_finite_spectrogram = TorchSigIterableDataset(
    dataset_metadata = dataset_finite_spectrogram_metadata,
    target_labels=["class_index"], #Gives you class index
    transforms = [Spectrogram(fft_size=fft_size)]
)

data, metadata = next(dataset_finite_spectrogram)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.imshow(data,aspect='auto',cmap='Wistia',vmin=0)
ax.set_xlabel('Time Axis')
ax.set_ylabel('Frequency Axis')
plt.show()


#------------------------------------------------
"""Saving dataset and writing to disk"""
#------------------------------------------------

root = "/home/vboxuser/MorganStateTorchSig/MSUTorch/generate_signal/group_example" #Path that the dataset is being held in
class_list = ["qpsk"] #Calling qpsk signals 
dataset_length = len(class_list) * 10 #Computing dataset_length of 10 qpsk signals #The original example uses all the signals but we are focusing on qpsk
seed = 123456789 

dataset_finite_metadata = DatasetMetadata(
    num_iq_samples_dataset = num_iq_samples_dataset,
    fft_size = fft_size,
    num_signals_max = num_signals_max,
    num_signals_min = num_signals_min,
)

# Don't use target_labels to get Signal objects with rich metadata
dataset = TorchSigIterableDataset(
    dataset_metadata = dataset_finite_metadata,
    transforms = [Spectrogram(fft_size = fft_size)],
    target_labels = ["class_name", "start", "stop", "lower_freq", "upper_freq", "snr_db"],
)

dataloader = WorkerSeedingDataLoader(dataset, batch_size=9, num_workers=1, collate_fn=default_collate_fn)
dataloader.seed(seed)

# New simplified DatasetCreator API # Used for organizing and storing large amounts of data
dataset_creator = DatasetCreator(
    dataset_length=dataset_length,
    dataloader = dataloader,
    root = root,
    overwrite = True,
    multithreading=False
)

dataset_creator.create()

#--------------------------
#Reading dataset from disk
#-------------------------
"""Static dataset is used for loading a saved dataset from disk. 
Samples can be accessed in any order and previously generated samples are accesible.
To then save a dataset to disk, use a torchsig.utils.writer.DatasetCreator which accepts a TorchSigIterableDataset object."""
static_dataset = StaticTorchSigDataset(
    root = root, 
    target_labels=dataset.target_labels
)

# can access any sample
print(static_dataset[0])
print(static_dataset[5])

# --------------------------
# Dataset Statistics
# --------------------------
static_dataset = TorchSigIterableDataset(
    dataset_metadata=dataset_metadata,
    target_labels=["class_name", "start", "stop", "lower_freq", "upper_freq", "snr_db"]
)

class_counter = {c: 0 for c in class_list}
snr_list = []
num_signals_per_sample = []

for sample in tqdm(static_dataset, desc="Calculating Dataset Stats"):
    data, targets = sample
    if isinstance(targets, tuple):
        num_signals_per_sample.append(1)
        classname, start, stop, lower, upper, snr = targets
        class_counter[classname] += 1
        snr_list.append(snr)
    elif isinstance(targets, list):
        num_signals_per_sample.append(len(targets))
        for t in targets:
            classname, start, stop, lower, upper, snr = t
            class_counter[classname] += 1
            snr_list.append(snr)

# Class Distribution Bar Chart
plt.figure(figsize=(12, 4))
plt.bar(class_counter.keys(), class_counter.values())
plt.xticks(rotation=45)
plt.xlabel("Modulation Class Name")
plt.ylabel("Counts")
plt.title("Class Distribution")
plt.show()

