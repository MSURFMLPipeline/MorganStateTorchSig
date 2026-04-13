class MSUDataBlock: #Initializing metadata
    def __init__ (self):
        pass
#----------------------------------------------------------------------------------------------------
#Trying to make functions that make rules for the different data options (EMI, Clean, EMI+Clean)
#This class passes the MetaData through and categorizes it into Clean, EMI, or Clean+EMI
#-----------------------------------------------------------------------------------------------------
    def Dataset_Mode(self):
        print(f"Dataset mode enforced: {mode}")
        print(f"Dataset root set to: {root}")

    def selection_conditions(self):
        if mode=="clean":
            self.snr_db_min=snr[mode]["snr_db_min"]
            self.snr_db_max=snr[mode]["snr_db_max"]
            self.class_list=class_lists[mode]
        elif mode=="emi":
            self.snr_db_min=snr[mode]["snr_db_min"]
            self.snr_db_max=snr[mode]["snr_db_max"]
            self.class_list=class_lists[mode]
        elif mode=="emi+clean":
            self.snr_db_min=snr[mode]["snr_db_min"]
            self.snr_db_max=snr[mode]["snr_db_max"]
            self.class_list=class_lists[mode]
        else:
            raise ValueError(f"The mode is not known: {mode}")

    
    def update_return_metadata(self): #Saving values within the Imported TorchSig Class
        self.dataset_metadata=DatasetMetadata(
            num_iq_samples_dataset=ds["num_iq_samples_dataset"],
            num_signals_min=ds["num_signals_min"],
            fft_size=ds["fft_size"],
            num_signals_max=ds["num_signals_max"],
            snr_db_min=self.snr_db_min,
            snr_db_max=self.snr_db_max,
            class_list=self.class_list,
            sample_rate=ds["sample_rate"],
            signal_bandwidth_freq_max=ds["signal_bandwidth_freq_max"],
            signal_bandwidth_freq_min=ds["signal_bandwidth_freq_min"],
            signal_center_freq_min=ds["signal_center_freq_min"],
            signal_center_freq_max=ds["signal_center_freq_max"],
            signal_duration_in_samples_min=ds["signal_duration_in_samples_min"],
            signal_duration_in_samples_max=ds["signal_duration_in_samples_max"],
            signal_bandwidth_max=ds["signal_bandwidth_max"],
            signal_bandwidth_min=ds["signal_bandwidth_min"],
            channel_freq_offset=0,
            phase_noise=0,
            timing_offset=0,
            fading=False,
        )
        print(f"Here are the {mode} metadata values that you entered for {self.dataset_metadata}:")
        return self.dataset_metadata #Returning Metadata

     #EMI + Noise Signals
    def metadata_tone(self):
          self.tone_metadata=DatasetMetadata(
            num_iq_samples_dataset=ds["num_iq_samples_dataset"],
            num_signals_min=ds["num_signals_min"],
            fft_size=ds["fft_size"],
            num_signals_max=ds["num_signals_max"],
            snr_db_min=self.snr_db_min,
            signal_bandwidth_max=ds["signal_bandwidth_max"],
            signal_bandwidth_min=ds["signal_bandwidth_min"],
            snr_db_max=self.snr_db_max,
            class_list=self.class_list,
            signal_bandwidth_freq_max=ds["signal_bandwidth_freq_max"],
            signal_bandwidth_freq_min=ds["signal_bandwidth_freq_min"],
            sample_rate=ds["sample_rate"],
            signal_center_freq_min=ds["signal_center_freq_min"],
            signal_center_freq_max=ds["signal_center_freq_max"],
            signal_duration_in_samples_min=ds["signal_duration_in_samples_min"],
            signal_duration_in_samples_max=ds["signal_duration_in_samples_max"],
            )
          print(f"Here are the {mode} metadata values that you entered for {self.tone_metadata}:")
          return self.tone_metadata #Returning Metadata
        
    
    def AWGN_noise_power(self):
        #Applying AWGN to clean signal
        #Enter desired power for noise(ranging from -30 db to 0 dB) in the config.yaml file
        self.noise=AWGN(noise_power_db=noise,bool=True)
        return self.noise
    
        
    def signal_impairments(self):
        self.impairments=Impairments(2) #Wireless Environment                                                
        self.burst_impairments=self.impairments.signal_transforms
        self.whole_signal_impairments=self.impairments.dataset_transforms
        self.burst_impairments,self.whole_signal_impairments
        
    def signal_training_impaired_dataset(self):
        self.training_impaired_dataset = TorchSigIterableDataset(
            dataset_metadata=self.tone_metadata,
            transforms=[self.whole_signal_impairments,self.noise],
            component_transforms=[self.burst_impairments])
        
    def signal_training_clean_plus_impaired_dataset(self):
        self.training_impaired_dataset_2 = TorchSigIterableDataset(
            dataset_metadata=self.dataset_metadata,
            transforms=[self.whole_signal_impairments,self.noise],
            component_transforms=[self.burst_impairments])
    
    def signal_testing_impaired_dataset(self):
        self.testing_impaired_dataset= TorchSigIterableDataset(
            dataset_metadata=self.tone_metadata,
            transforms=[self.whole_signal_impairments,self.noise],
            component_transforms=[self.burst_impairments])
        
    def signal_validation_impaired_dataset(self):
        self.validation_impaired_dataset= TorchSigIterableDataset(
            dataset_metadata=self.tone_metadata,
            transforms=[self.whole_signal_impairments,self.noise],
            component_transforms=[self.burst_impairments])
     
    def creating_noisy_validation_dataloader(self):
        self.noisy_validation_dataloader=WorkerSeedingDataLoader(self.validation_impaired_dataset,collate_fn=lambda x:x)
        self.noisy_validation_dataloader.seed(ds["seed"])
    
    def creating_noisy_testing_dataloader(self):
        self.noisy_testing_dataloader=WorkerSeedingDataLoader(self.testing_impaired_dataset,collate_fn=lambda x:x)
        self.noisy_testing_dataloader.seed(ds["seed"])
    
    def creating_noisy_training_dataloader(self):
        self.noisy_training_dataloader=WorkerSeedingDataLoader(self.training_impaired_dataset,collate_fn=lambda x:x)
        self.noisy_training_dataloader.seed(ds["seed"])
    
    def writing_noisy_validation_dataset(self):
        #writes the workerseedingdataloader dataset to disk
        self.num_noisy_val_samples = len(self.class_list) * 500 # roughly 50 samples per class
        print(f"The number of noisy training samples is {self.num_noisy_val_samples}")

        dc = DatasetCreator(
        dataloader=self.noisy_validation_dataloader,
        root = f"{root}/noise_validation",
        overwrite=True,
        dataset_length=self.num_noisy_val_samples)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:",f"{root}/emi/noise_validation")
    
    def writing_noisy_testing_dataset(self):
        #writes the workerseedingdataloader dataset to disk
        self.num_noisy_test_samples = len(self.class_list) * 400 # roughly 50 samples per class
        print(f"The number of noisy training samples is {self.num_noisy_test_samples}")

        dc = DatasetCreator(
        dataloader=self.noisy_testing_dataloader,
        root = f"{root}/noise_testing",
        overwrite=True,
        dataset_length=self.num_noisy_test_samples)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:",f"{root}/noise_testing")
        
    def writing_noisy_training_dataset(self):
        #writes the workerseedingdataloader dataset to disk
        self.num_noisy_train_samples = len(self.class_list) * 100 # roughly 50 samples per class
        print(f"The number of noisy training samples is {self.num_noisy_train_samples}")

        dc = DatasetCreator(
        dataloader=self.noisy_training_dataloader,
        root = f"{root}/noise_training",
        overwrite=True,
        dataset_length=self.num_noisy_train_samples)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:",f"{root}/noise_training")
    
    def reading_noisy_training_dataset(self):
        #Reads the dataset from disk
        self.noise_training_static_dataset=StaticTorchSigDataset(
            root=f"{root}/noise_training",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"]
            )
        print("\nLoaded noise training static dataset length:", len(self.noise_training_static_dataset))
        print(self.noise_training_static_dataset)
    
    def reading_noisy_testing_dataset(self):
        #Reads the dataset from disk
        self.noise_testing_static_dataset=StaticTorchSigDataset(
            root=f"{root}/noise_testing",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"],
            )
        print("\nLoaded noise testing static dataset length:", len(self.noise_testing_static_dataset))
        print(self.noise_testing_static_dataset)
        
    def reading_noisy_validation_dataset(self):
        #Reads the dataset from disk
        self.noise_validation_static_dataset=StaticTorchSigDataset(
            root=f"{root}/noise_validation",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"],
            )
        print("\nLoaded noise testing static dataset length:", len(self.noise_validation_static_dataset))
        print(self.noise_validation_static_dataset)

    def inspecting_noisy_validation_batch(self):
        self.val_noise_spectrograms, self.val_noise_labels = self.noise_training_static_dataset[1]
        print("\n---Test Spectrogram INFO ---")
        print("\nTest Spectrogram batch shape:", self.val_noise_spectrograms.shape)
        print(f"\nData type: {self.val_noise_spectrograms.dtype}")
        print(f"\nValidation YOlO Labels (Class Index, X center, Y center, Width, Height):", self.val_noise_labels)
        print("Spectrogram (Real) I Samples:", self.val_noise_spectrograms.real)
        print("Spectrogram (Imaginary) Q Samples:", self.val_noise_spectrograms.imag)

    def inspecting_noisy_testing_batch(self):
        self.test_noise_spectrograms, self.test_noise_labels = self.noise_testing_static_dataset[1]
        print("\n---Test Spectrogram INFO ---")
        print("\nTest Spectrogram batch shape:", self.test_noise_spectrograms.shape)
        print(f"\nData type: {self.test_noise_spectrograms.dtype}")
        print(f"\nValidation YOlO Labels (Class Index, X center, Y center, Width, Height):", self.test_noise_labels)
        print("Spectrogram (Real) I Samples:", self.test_noise_spectrograms.real)
        print("Spectrogram (Imaginary) Q Samples:", self.test_noise_spectrograms.imag)
        
    def inspecting_noisy_training_batch(self):
        self.train_noise_spectrograms, self.train_noise_labels = self.noise_training_static_dataset[1]
        print("\n---Test Spectrogram INFO ---")
        print("\nTest Spectrogram batch shape:", self.train_noise_spectrograms.shape)
        print(f"\nData type: {self.train_noise_spectrograms.dtype}")
        print(f"\nValidation YOlO Labels (Class Index, X center, Y center, Width, Height):", self.train_noise_labels)
        print("Spectrogram (Real) I Samples:", self.train_noise_spectrograms.real)
        print("Spectrogram (Imaginary) Q Samples:", self.train_noise_spectrograms.imag)
   
    def noisy_spectrogram_directories(self):
        os.makedirs(disk_root, exist_ok=True)
        self.noise_train_label_dir = f"{disk_root}/labels/emi/train"
        self.noise_train_image_dir = f"{disk_root}/images/emi/train"
        os.makedirs(self.noise_train_label_dir, exist_ok=True)
        os.makedirs(self.noise_train_image_dir, exist_ok=True)

        self.noise_val_label_dir = f"{disk_root}/labels/emi/val"
        self.noise_val_image_dir = f"{disk_root}/images/emi/val"
        os.makedirs(self.noise_val_label_dir, exist_ok=True)
        os.makedirs(self.noise_val_image_dir, exist_ok=True)

        self.noise_test_label_dir = f"{disk_root}/labels/emi/test"
        self.noise_test_image_dir = f"{disk_root}/images/emi/test"
        os.makedirs(self.noise_test_label_dir, exist_ok=True)
        os.makedirs(self.noise_test_image_dir, exist_ok=True)
