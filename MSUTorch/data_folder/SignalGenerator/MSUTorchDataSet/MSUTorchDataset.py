"""Generating MSUTORCH CLASSES"""
import numpy as np
class MSUTorchDataset:
    def __init__(self,sample_rate,iq_samples,fft_size,snr_min,snr_max,min_signals,max_signals,class_index):
        self.sample_rate=sample_rate
        self.iq_samples=iq_samples
        self.fft_size=fft_size
        self.snr_min=snr_min
        self.snr_max=snr_max
        self.min_signals=min_signals
        self.max_signals=max_signals
        self.class_index=class_index

        