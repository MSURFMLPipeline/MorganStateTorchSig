"""Generating MSUTORCH CLASSES"""
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

class MSUTorchMetaData:
    def __init__(self,num_iq_samples_dataset, fft_size, num_signals_min, num_signals_max, snr_db_min, snr_db_max, class_list) 

    self.num_iq_samples_dataset=num_iq_samples_dataset
    self.fft_size=fft_size
    self.num_signals_min=num_signals_min
    self.num_signals_max=num_signals_max
    self.snr_db_min=snr_db_min
    self.snr_db_max=snr_db_max
    self.class_list=class_list

    def MSUTorchMetaData_to_TorchSigMetadata
     return DatasetMetadata(
        num_iq_samples_dataset=self.num_iq_samples_dataset,
        fft_size=self.fft_size,
        num_signals_min=self.num_signals_min,
        num_signals_max=self.num_signals_max,
        snr_db_min=self.snr_db_min,
        snr_db_max=self.snr_db_max,
        class_list=self.class_list,

class MSUTorchDataset(MSUTorchMetaData):
    def __init__(self): #Inherits from self 
        super().__init__()

    def dataset(self):
        return self
        
     def clean_data(self):
        if self.snr_min != self.snr_max:
            return print("The signal will not be a clean singal")
        elif self.snr_min == self.snr_max ==30:
            print("The signal is clean")
            return self.snr_max
        else:
            return self.snr_min,self.snr_max

#    def iq_data(self):
        #Splitting into real and imaginary

 #   def EMI_data(self):
        #EMI data or tones

#    class ComponentSignals


"""
        dataset_time_series = TorchSigIterableDataset(
dataset_metadata = dataset_time_series_metadata, 
target_labels=["class_index"] """

metadata=MSUTorchDataset(90,10, 20, 10, 39, 40, 70, 9)
metadata.clean()
