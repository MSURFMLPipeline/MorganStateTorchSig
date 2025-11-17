import matplotlib.pyplot as plt
from MSUTorch.data_folder.datasets import DatasetMetadata
from MSUTorch.data_folder.metadata import TorchSigIterableDataset

#Define dataset parameters

num_iq_samples_dataset = 1024
fft_size = 128
num_signals_min = 1
num_signals_max = 1
snr_db_min = 10
snr_db_max = 30


#Create dataset metadata

dataset_metadata = DatasetMetadata(
    num_iq_samples_dataset=num_iq_samples_dataset,
    fft_size=fft_size,
    num_signals_min=num_signals_min,
    num_signals_max=num_signals_max,
    snr_db_min=snr_db_min,
    snr_db_max=snr_db_max
)

#Create iterable dataset

dataset = TorchSigIterableDataset(dataset_metadata=dataset_metadata)


#Generate a signal9-0-9-080

signal = next(dataset)  # get first synthetic signal
iq_data = signal.data    # IQ samples (already a NumPy array)

#Plot spectrogram of the real part

plt.figure(figsize=(10, 4))
plt.specgram(iq_data.real, NFFT=fft_size, Fs=1.0, noverlap=fft_size//2, cmap='viridis')
plt.title('Spectrogram of Synthetic IQ Signal (Real Part)')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label='Intensity [dB]')
plt.show()



