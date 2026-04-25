import numpy as np
import matplotlib.pyplot as plt
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.datasets import TorchSigIterableDataset

#QPSK Data
num_iq_samples_dataset = 8192
fft_size = 256
num_signals_min = 1
num_signals_max = 1
snr_db_min = 25
snr_db_max = 30
class_list = ["qpsk"]


# Dataset Metadata

dataset_metadata = DatasetMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    fft_size=fft_size,
    num_signals_min=num_signals_min,
    num_signals_max=num_signals_max,
    snr_db_min=snr_db_min,
    snr_db_max=snr_db_max,
    class_list=class_list,
)


# Create Dataset

dataset = TorchSigIterableDataset(dataset_metadata=dataset_metadata)

# Generate one QPSK signal
signal = next(dataset) #Python Iterator
iq_data = signal.data  # Gets the complex IQ samples thats in the form of an array #.data is used to get the raw data
print(signal.data)
print(signal.data.shape)

# Spectrogram Plot (Magnitude)

plt.figure(figsize=(10, 4))
plt.specgram(np.abs(iq_data), NFFT=fft_size,Fs=1.0,noverlap=fft_size // 2, cmap="rainbow")
""" np.abs is the absolute value of iq_data
NFFT is the set FFT size for each segment
noverlap is telling how many points overlap so in are case it would be 128
cmap is the color map"""
plt.title("Spectrogram of Synthetic QPSK Signal")
plt.xlabel("Time [seconds]")
plt.ylabel("Frequency")
plt.colorbar(label="Power [dB]")
plt.tight_layout() #Used to avoid clipping
plt.show()