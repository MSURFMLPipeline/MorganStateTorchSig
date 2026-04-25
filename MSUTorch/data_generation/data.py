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

    #Spliting data into testing and validation (Finished)
    def training_dataset(self):
        self.train_dataset = TorchSigIterableDataset(self.dataset_metadata,seed=train["seed"],signal_generators=self.signal_builders)
        #print(self.train_samples)

    def validation_dataset(self):
        self.val_dataset = TorchSigIterableDataset(self.dataset_metadata,seed=validation["seed"],signal_generators=self.signal_builders)

    def testing_dataset(self):
        self.test_dataset=TorchSigIterableDataset(self.dataset_metadata,seed=test["seed"],signal_generators=self.signal_builders)

    def creating_testing_dataloader(self):
        self.test_dataloader=WorkerSeedingDataLoader(self.test_dataset,batch_size=test["batch_size"], collate_fn=lambda x:x)
        self.test_dataloader.seed(ds["seed"])

    def creating_training_dataloader(self):
        #Contains the dataset that we created for training data

        self.train_dataloader = WorkerSeedingDataLoader(self.train_dataset, batch_size=train["batch_size"], collate_fn=lambda x:x)
        self.train_dataloader.seed(ds["seed"])

    def creating_validation_dataloader(self):
        self.val_dataloader = WorkerSeedingDataLoader(self.val_dataset, batch_size=validation["batch_size"], collate_fn = lambda x: x)
        self.val_dataloader.seed(ds["seed"])

    def writing_testing_dataset(self):
        #writes the workerseedingdataloader dataset to disk
        self.num_samples_test = len(self.class_list) * 800 # roughly 100 samples per class
        print(f"The number of testing samples is {self.num_samples_test}")

        dc = DatasetCreator(
        dataloader=self.test_dataloader,
        root = f"{root}/test",
        overwrite=True,
        dataset_length=self.num_samples_test)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:", f"{root}/test")

    def writing_training_dataset(self):

        #writes the workerseedingdataloader dataset to disk
        self.num_samples_train = len(self.class_list) * 100  # roughly 700 samples per class
        print(f"The number of training samples is {self.num_samples_train}")

        dc = DatasetCreator(
        dataloader=self.train_dataloader,
        root = f"{root}/train",
        overwrite=True,
        dataset_length=self.num_samples_train)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:",f"{root}/train")

    def writing_validation_dataset(self):
        self.num_samples_val = len(self.class_list) * 100  #roughly 200 samples per class
        print(f"The number of validation samples is {self.num_samples_val}")

        dc = DatasetCreator(
        dataloader=self.val_dataloader,
        root = f"{root}/val",
        overwrite=True,
        dataset_length=self.num_samples_val)

        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:", f"{root}/val")

    def reading_testing_dataset(self):
           #Reads the dataset from disk
        self.testing_static_dataset=StaticTorchSigDataset(
            root=f"{root}/test",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"]
            )
        print("\nLoaded testing static dataset length:", len(self.testing_static_dataset))
        print(self.testing_static_dataset)

    def reading_training_dataset(self):
        #Reads the dataset from disk
        self.training_static_dataset=StaticTorchSigDataset(
            root=f"{root}/train",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"]
            )
        print("\nLoaded training static dataset length:", len(self.training_static_dataset))
        print(self.training_static_dataset)

    def reading_validation_dataset(self):
        #Reads the dataset from disk
        self.validation_static_dataset=StaticTorchSigDataset(
            root=f"{root}/val",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"]
            )

        print("\nLoaded validation static dataset length:", len(self.validation_static_dataset))
        print(self.validation_static_dataset)

    def inspecting_test_batch(self):
        self.test_spectrograms, self.test_labels = self.testing_static_dataset[1]
        print("\n---Test Spectrogram INFO ---")
        print("\nTest Spectrogram batch shape:", self.test_spectrograms.shape)
        print(f"\nData type: {self.test_spectrograms.dtype}")
        print(f"\nValidation YOlO Labels (Class Index, X center, Y center, Width, Height):", self.test_labels)
        print("Spectrogram (Real) I Samples:", self.test_spectrograms.real)
        
        
    def inspecting_validation_batch(self):
        self.spectrograms, self.val_labels = self.validation_static_dataset[1]
        print("\n---Validation Spectrogram INFO ---")
        print("\nValidation Spectrogram batch shape:", self.spectrograms.shape)
        print(f"\nData type: {self.spectrograms.dtype}")
        print(f"\nValidation YOlO Labels (Class Index, X center, Y center, Width, Height):", self.val_labels)
        print("Spectrogram (Real) I Samples:", self.spectrograms.real)
        
    def inspecting_training_batch(self):
        self.train_spec_data, self.labels = self.training_static_dataset[1]
        print("\n---Training Spectrogram INFO ---")
        print("\nTraining Spectrogram batch shape:", self.train_spec_data.shape)
        print(f"\nData type: {self.train_spec_data.dtype} ")
        print(f"\nTraining YOlO Labels (Class Index, X center, Y center, Width, Height):", self.labels)
        print("Spectrogram (Real) I Samples:", self.train_spec_data.real)
        
    def spectrogram_image_directories(self):
        os.makedirs(disk_root, exist_ok=True)
        self.label_dir = f"{disk_root}/labels/clean/train"
        self.image_dir = f"{disk_root}/images/clean/train"
        os.makedirs(self.label_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)


        self.val_label_dir = f"{disk_root}/labels/clean/val"
        self.val_image_dir = f"{disk_root}/images/clean/val"
        os.makedirs(self.val_label_dir, exist_ok=True)
        os.makedirs(self.val_image_dir, exist_ok=True)

        self.test_label_dir = f"{disk_root}/labels/clean/test"
        self.test_image_dir = f"{disk_root}/images/clean/test"
        os.makedirs(self.test_label_dir, exist_ok=True)
        os.makedirs(self.test_image_dir, exist_ok=True)

    def writing_YOLO_Testing_Dataset(self):
# tqdm adds progress bars so that we can see the static training dataset being written as labels and images"

        for i, (test_data, test_spec_labels) in tqdm(
            enumerate(self.testing_static_dataset),
            total=len(self.testing_static_dataset),
            desc="\nWriting YOLO Test Dataset"):

            file_base = str(i).zfill(10)
            test_label_file = f"{self.test_label_dir}/{file_base}.txt"
            test_img_file = f"{self.test_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            test_img = test_data.detach().numpy() if hasattr(test_data, 'detach') else np.array(test_data)
            test_img = np.squeeze(test_img).astype(np.float32)
            test_img = np.abs(test_img)
            test_img = test_img - test_img.min()
            test_img = np.maximum(test_img, 1e-12)
            test_img = np.nan_to_num(test_img, nan=0.0, posinf=0.0, neginf=0.0)

            if test_img.max() > 0:
                test_img = test_img / test_img.max()
                test_img = (test_img * 255).astype(np.uint8)
                test_img = np.stack([test_img, test_img, test_img], axis=-1)
                Image.fromarray(test_img).save(test_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(test_label_file, "w") as f:
                if len(test_spec_labels) == 0:
                    pass
                if len(test_spec_labels) > 0 and isinstance(test_spec_labels[0], (list, tuple, np.ndarray)):
                    for l in test_spec_labels:
                       f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
                elif len(test_spec_labels) > 0:
                    l = test_spec_labels
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")


    print("\nYOLO Test Dataset Generation Complete.")


    def writing_YOLO_Validation_Dataset(self):
        for i, (val_data, val_spec_labels) in tqdm(
            enumerate(self.validation_static_dataset),
            total=len(self.validation_static_dataset),
            desc="\nWriting YOLO Validation Dataset"):

            file_base=str(i).zfill(10)

            val_label_file = f"{self.val_label_dir}/{file_base}.txt"
            val_img_file = f"{self.val_image_dir}/{file_base}.png"

        # Convert spectrogram to image
            val_img = val_data.detach().numpy() if hasattr(val_data, 'detach') else np.array(val_data)
            val_img = np.squeeze(val_img).astype(np.float32)
            val_img = np.abs(val_img)
            val_img = val_img - val_img.min()
            val_img = np.maximum(val_img, 1e-12)
            val_img = np.nan_to_num(val_img, nan=0.0, posinf=0.0, neginf=0.0)

            if val_img.max() > 0:
                val_img = val_img / val_img.max()
                val_img = (val_img * 255).astype(np.uint8)
                val_img = np.stack([val_img, val_img, val_img], axis=-1)
                Image.fromarray(val_img).save(val_img_file)

        # Write YOLO label
            with open(val_label_file, "w") as f:
              if len(val_spec_labels) == 0:
                  pass
              if len(val_spec_labels) > 0 and isinstance(val_spec_labels[0], (list, tuple, np.ndarray)):
                  for l in val_spec_labels:
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
              elif len(val_spec_labels) > 0:
                  l = val_spec_labels
                  f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")


    print("\nYOLO Validation Dataset Generation Complete.")



    def writing_YOLO_Training_Dataset(self):
# tqdm adds progress bars so that we can see the static training dataset being written as labels and images"

        for i, (data, spec_labels) in tqdm(
            enumerate(self.training_static_dataset),
            total=len(self.training_static_dataset),
            desc="\nWriting YOLO Training Dataset"):
            
            file_base = str(i).zfill(10)
            label_file = f"{self.label_dir}/{file_base}.txt"
            img_file = f"{self.image_dir}/{file_base}.png"
        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            img = data.detach().numpy() if hasattr(data, 'detach') else np.array(data)
            img = np.squeeze(img).astype(np.float32)
            img = np.abs(img)
            img = np.log10(img + 1e-12)
            img = img - img.min()
            img = np.maximum(img, 1e-12)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            
            if img.max() > 0:
                img = img / img.max()
                img = (img * 255).astype(np.uint8)
                img = np.stack([img, img, img], axis=-1)
                Image.fromarray(img).save(img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(label_file, "w") as f:
                if len(spec_labels) > 0 and isinstance(spec_labels[0], (list, tuple, np.ndarray)):
                    for l in spec_labels:
                       f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
                elif len(spec_labels) > 0:
                    l = spec_labels
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
                        
        print("\nYOLO Training Dataset Generation Complete.")


#--------------------------------
     #EMI + Noise Signals
#--------------------------------
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
   
#-----------------------
#Clean+EMI Pipeline
#----------------------
    def metadata_clean_plus_emi(self):
        self.clean_plus_emi_metadata=DatasetMetadata(
            num_iq_samples_dataset=ds["num_iq_samples_dataset"],
            num_signals_min=self.num_signals_min,
            fft_size=ds["fft_size"],
            num_signals_max=self.num_signals_max,
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
        print(f"Here are the {mode} metadata values that you entered for {self.clean_plus_emi_metadata}:")
 
        return self.clean_plus_emi_metadata #Returning Clean+EMI Metadata        
    
    def AWGN_noise_power(self):
        #Applying AWGN to clean signal
        #Enter desired power for noise(ranging from -30 db to 0 dB) in the config.yaml file
        self.noise=AWGN(noise_power_db=noise)
        return self.noise
    
        
    def signal_impairments(self):
        self.impairments=Impairments(2) #Wireless Environment                                                
        self.burst_impairments=self.impairments.signal_transforms
        self.whole_signal_impairments=self.impairments.dataset_transforms
    
    def signal_generators_for_dataset(self):
        self.signal_builders=(TorchSigSignalLists.fsk_signals + TorchSigSignalLists.constellation_signals + TorchSigSignalLists.am_signals + TorchSigSignalLists.fm_signals)
    
    def signal_training_clean_plus_impaired_dataset(self):
        self.training_dataset_2 = TorchSigIterableDataset(
            dataset_metadata=self.clean_plus_emi_metadata,
            transforms=[self.whole_signal_impairments,self.noise],
            signal_generators=self.signal_builders+[ToneSignalBuilder(self.tone_metadata)])
        
    def signal_testing_clean_plus_impaired_dataset(self):
        self.testing_dataset_2 = TorchSigIterableDataset(
            dataset_metadata=self.clean_plus_emi_metadata,
            transforms=[self.whole_signal_impairments,self.noise],
            signal_generators=self.signal_builders+[ToneSignalBuilder(self.tone_metadata)]
            )
    
    def signal_validation_clean_plus_impaired_dataset(self):
        self.validation_dataset_2 = TorchSigIterableDataset(
            dataset_metadata=self.clean_plus_emi_metadata,
            transforms=[self.whole_signal_impairments,self.noise],
            signal_generators=self.signal_builders+[ToneSignalBuilder(self.tone_metadata)] 
            )
    
    def creating_training_cpe_training_dataloader(self):
        self.training_2_dataloader = WorkerSeedingDataLoader(self.training_dataset_2, collate_fn=lambda x:x)
        self.training_2_dataloader.seed(ds["seed"])
    def creating_testing_cpe_training_dataloader(self):
        self.testing_2_dataloader = WorkerSeedingDataLoader(self.testing_dataset_2, collate_fn=lambda x:x)
        self.testing_2_dataloader.seed(ds["seed"])
    def creating_validation_cpe_training_dataloader(self):
        self.validation_2_dataloader = WorkerSeedingDataLoader(self.validation_dataset_2, collate_fn=lambda x:x)
        self.validation_2_dataloader.seed(ds["seed"])
    def writing_training_cpe_dataset(self): 
        #writes the workerseedingdataloader dataset to disk
        self.num_training_cpe_samples = len(self.class_list) * 700 # roughly 50 samples per class
        print(f"The number of clean+emi training samples is {self.num_training_cpe_samples}")

        dc = DatasetCreator(
        dataloader=self.training_2_dataloader,
        root = f"{root}/training_clean_plus_emi",
        overwrite=True,
        dataset_length=self.num_training_cpe_samples)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:",f"{root}/clean_plus_emi/training_clean_plus_emi")

    def writing_testing_cpe_dataset(self): 
        #writes the workerseedingdataloader dataset to disk
        self.num_testing_cpe_samples = len(self.class_list) * 100 # roughly 50 samples per class
        print(f"The number of clean+emi testing samples is {self.num_testing_cpe_samples}")

        dc = DatasetCreator(
        dataloader=self.testing_2_dataloader,
        root = f"{root}/testing_clean_plus_emi",
        overwrite=True,
        dataset_length=self.num_testing_cpe_samples)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:",f"{root}/clean_plus_emi/testing_clean_plus_emi")
   
    def writing_validation_cpe_dataset(self): 
        #writes the workerseedingdataloader dataset to disk
        self.num_validation_cpe_samples = len(self.class_list) * 200 # roughly 50 samples per class
        print(f"The number of clean+emi validation samples is {self.num_validation_cpe_samples}")

        dc = DatasetCreator(
        dataloader=self.validation_2_dataloader,
        root = f"{root}/validation_clean_plus_emi",
        overwrite=True,
        dataset_length=self.num_validation_cpe_samples)


        print("\nWriting dataset to disk...")
        dc.create()
        print("Dataset written to:",f"{root}/clean_plus_emi/validation_clean_plus_emi")       

    def reading_cpe_training_dataset(self):
        #Reads the dataset from disk
        self.training_cpe_static_dataset=StaticTorchSigDataset(
            root=f"{root}/training_clean_plus_emi",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"]
            )
        print("\nLoaded training Clean+EMI static dataset length:", len(self.training_cpe_static_dataset))
        print(self.training_cpe_static_dataset)
    
    def reading_cpe_testing_dataset(self):
        #Reads the dataset from disk
        self.testing_cpe_static_dataset=StaticTorchSigDataset(
            root=f"{root}/testing_clean_plus_emi",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"]
            )
        print("\nLoaded testing Clean+EMI static dataset length:", len(self.testing_cpe_static_dataset))
        print(self.testing_cpe_static_dataset)
    
    def reading_cpe_validation_dataset(self):
        #Reads the dataset from disk
        self.validation_cpe_static_dataset=StaticTorchSigDataset(
            root=f"{root}/validation_clean_plus_emi",
            transforms=[Spectrogram(fft_size=ds["fft_size"]),YOLOLabel()],
            target_labels=["yolo_label"]
            )
        print("\nLoaded validation Clean+EMI static dataset length:", len(self.validation_cpe_static_dataset))
        print(self.validation_cpe_static_dataset)
    
    def inspecting_cpe_validation_batch(self):
        self.val_cpe_spectrograms, self.val_cpe_labels = self.validation_cpe_static_dataset[1]
        print("\n---Validation Spectrogram INFO ---")
        print("\nValidation Spectrogram batch shape:", self.val_cpe_spectrograms.shape)
        print(f"\nData type: {self.val_cpe_spectrograms.dtype}")
        print(f"\nValidation YOlO Labels (Class Index, X center, Y center, Width, Height):", self.val_cpe_labels)
        print("Spectrogram (Real) I Samples:", self.val_cpe_spectrograms.real)
        
    def inspecting_cpe_testing_batch(self):
        self.test_cpe_spectrograms, self.test_cpe_labels = self.testing_cpe_static_dataset[1]
        print("\n---Test Spectrogram INFO ---")
        print("\nTest Spectrogram batch shape:", self.test_cpe_spectrograms.shape)
        print(f"\nData type: {self.test_cpe_spectrograms.dtype}")
        print(f"\nValidation YOlO Labels (Class Index, X center, Y center, Width, Height):", self.test_cpe_labels)
        print("Spectrogram (Real) I Samples:", self.test_cpe_spectrograms.real)
    

    def inspecting_cpe_training_batch(self):
        self.train_cpe_spectrograms, self.train_cpe_labels = self.training_cpe_static_dataset[1]
        print("\n---Training Spectrogram INFO ---")
        print("\nTraining Spectrogram batch shape:", self.train_cpe_spectrograms.shape)
        print(f"\nData type: {self.train_cpe_spectrograms.dtype}")
        print(f"\nTraining YOlO Labels (Class Index, X center, Y center, Width, Height):", self.train_cpe_labels)
        print("Spectrogram (Real) I Samples:", self.train_cpe_spectrograms.real)    
    
    def cpe_spectrogram_directories(self):
        os.makedirs(disk_root, exist_ok=True)
        self.clean_emi_train_label_dir = f"{disk_root}/labels/clean+emi/train"
        self.clean_emi_train_image_dir = f"{disk_root}/images/clean+emi/train"
        os.makedirs(self.clean_emi_train_label_dir, exist_ok=True)
        os.makedirs(self.clean_emi_train_image_dir, exist_ok=True)

        self.clean_emi_val_label_dir = f"{disk_root}/labels/clean+emi/val"
        self.clean_emi_val_image_dir = f"{disk_root}/images/clean+emi/val"
        os.makedirs(self.clean_emi_val_label_dir, exist_ok=True)
        os.makedirs(self.clean_emi_val_image_dir, exist_ok=True)

        self.clean_emi_test_label_dir = f"{disk_root}/labels/clean+emi/test"
        self.clean_emi_test_image_dir = f"{disk_root}/images/clean+emi/test"
        os.makedirs(self.clean_emi_test_label_dir, exist_ok=True)
        os.makedirs(self.clean_emi_test_image_dir, exist_ok=True)
        
    def writing_YOLO_CPE_Test_Dataset(self):

        for i, (test_clean_emi_data, test_clean_emi_labels) in tqdm(
            enumerate(self.testing_cpe_static_dataset),
            total=len(self.testing_cpe_static_dataset),
            desc="\nWriting YOLO Testing Dataset"):

            file_base = str(i).zfill(10)
            test_clean_emi_label_file = f"{self.clean_emi_test_label_dir}/{file_base}.txt"
            test_clean_emi_img_file = f"{self.clean_emi_test_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            test_clean_emi_img = test_clean_emi_data.detach().cpu().numpy() if hasattr(test_clean_emi_data, 'detach') else np.array(test_clean_emi_data)
            test_clean_emi_img = np.squeeze(test_clean_emi_img).astype(np.float32)
            test_clean_emi_img = np.abs(test_clean_emi_img)
            test_clean_emi_img = np.log10(test_clean_emi_img + 1e-12) 
            test_clean_emi_img = test_clean_emi_img - test_clean_emi_img.min()
            test_clean_emi_img = np.nan_to_num(test_clean_emi_img, nan=0.0, posinf=0.0, neginf=0.0)
            test_clean_emi_img = np.maximum(test_clean_emi_img, 1e-12)

            if test_clean_emi_img.max() > 0:
                test_clean_emi_img = test_clean_emi_img / test_clean_emi_img.max()
                test_clean_emi_img = (test_clean_emi_img * 255).astype(np.uint8)
                test_clean_emi_img = np.stack([test_clean_emi_img, test_clean_emi_img, test_clean_emi_img], axis=-1)
                Image.fromarray(test_clean_emi_img).save(test_clean_emi_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(test_clean_emi_label_file, "w") as f:
                if isinstance(test_clean_emi_labels[0], (list, tuple, np.ndarray)):
                    for l in test_clean_emi_labels:
                        f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")
                else:
                    l = test_clean_emi_labels
                    f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")

    print("\nYOLO Testing Clean+EMI Dataset Generation Complete.")


    def writing_YOLO_CPE_Validation_Dataset(self):

        for i, (val_clean_emi_data, val_clean_emi_labels) in tqdm(
            enumerate(self.validation_cpe_static_dataset),
            total=len(self.validation_cpe_static_dataset),
            desc="\nWriting YOLO Training Dataset"):

            file_base = str(i).zfill(10)
            val_clean_emi_label_file = f"{self.clean_emi_val_label_dir}/{file_base}.txt"
            val_clean_emi_img_file = f"{self.clean_emi_val_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            val_clean_emi_img = val_clean_emi_data.detach().cpu().numpy() if hasattr(val_clean_emi_data, 'detach') else np.array(val_clean_emi_data)
            val_clean_emi_img = np.squeeze(val_clean_emi_img).astype(np.float32)
            val_clean_emi_img = np.abs(val_clean_emi_img)
            val_clean_emi_img = np.log10(val_clean_emi_img + 1e-12)
            val_clean_emi_img = val_clean_emi_img - val_clean_emi_img.min()
            val_clean_emi_img = np.maximum(val_clean_emi_img, 1e-12)
            val_clean_emi_img = np.nan_to_num(val_clean_emi_img, nan=0.0, posinf=0.0, neginf=0.0)

            if val_clean_emi_img.max() > 0:
                val_clean_emi_img = val_clean_emi_img / val_clean_emi_img.max()
                val_clean_emi_img = (val_clean_emi_img * 255).astype(np.uint8)
                val_clean_emi_img = np.stack([val_clean_emi_img, val_clean_emi_img, val_clean_emi_img], axis=-1)
                Image.fromarray(val_clean_emi_img).save(val_clean_emi_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(val_clean_emi_label_file, "w") as f:
                if isinstance(val_clean_emi_labels[0], (list, tuple, np.ndarray)):
                    for l in val_clean_emi_labels:
                        f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")
                else:
                    l = val_clean_emi_labels
                    f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")

    print("\nYOLO Validation Clean+EMI Dataset Generation Complete.")
    
    def writing_YOLO_CPE_Training_Dataset(self):

        for i, (train_clean_emi_data, train_clean_emi_labels) in tqdm(
            enumerate(self.training_cpe_static_dataset),
            total=len(self.training_cpe_static_dataset),
            desc="\nWriting YOLO Training Dataset"):

            file_base = str(i).zfill(10)
            train_clean_emi_label_file = f"{self.clean_emi_train_label_dir}/{file_base}.txt"
            train_clean_emi_img_file = f"{self.clean_emi_train_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            train_clean_emi_img = train_clean_emi_data.detach().numpy() if hasattr(train_clean_emi_data, 'detach') else np.array(train_clean_emi_data)
            train_clean_emi_img = np.squeeze(train_clean_emi_img).astype(np.float32)
            train_clean_emi_img = np.abs(train_clean_emi_img)
            train_clean_emi_img = np.log10(train_clean_emi_img + 1e-12)
            train_clean_emi_img = train_clean_emi_img - train_clean_emi_img.min()
            train_clean_emi_img = np.maximum(train_clean_emi_img, 1e-12)
            train_clean_emi_img = np.nan_to_num(train_clean_emi_img, nan=0.0, posinf=0.0, neginf=0.0)

            if train_clean_emi_img.max() > 0:
                train_clean_emi_img = train_clean_emi_img / train_clean_emi_img.max()
                train_clean_emi_img = (train_clean_emi_img * 255).astype(np.uint8)
                train_clean_emi_img = np.stack([train_clean_emi_img, train_clean_emi_img, train_clean_emi_img], axis=-1)
                Image.fromarray(train_clean_emi_img).save(train_clean_emi_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(train_clean_emi_label_file, "w") as f:
                if isinstance(train_clean_emi_labels[0], (list, tuple, np.ndarray)):
                    for l in train_clean_emi_labels:
                        f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")
                else:
                    l = train_clean_emi_labels
                    f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")

    print("\nYOLO Training Clean+EMI Dataset Generation Complete.")  