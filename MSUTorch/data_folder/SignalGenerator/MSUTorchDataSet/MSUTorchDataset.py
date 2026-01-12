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

class MSUTorchDataset:
    def __init__(self,sample_rate,iq_samples,fft_size,snr_min,snr_max,min_signals,max_signals,class_index):
           super().__init
        self.sample_rate=sample_rate
        self.iq_samples=iq_samples
        self.fft_size=fft_size
        self.snr_min=snr_min
        self.snr_max=snr_max
        self.min_signals=min_signals
        self.max_signals=max_signals
        self.class_index=class_index

    def modulation(self):
    
        
     def clean_data(self):
        if self.snr_min != self.snr_max:
            return print("The signal will not be a clean singal")
        elif self.snr_min&self.snr_max==30:
            print("The signal is clean")
            return self.snr_max
        else:
            return self.snr_min,self.snr_max



"""
        dataset_time_series = TorchSigIterableDataset(
dataset_metadata = dataset_time_series_metadata, 
target_labels=["class_index"] """

s1=MSUTorchDataset(90,10, 20, 10, 39, 40, 70, 9 )
s1.modulation()
