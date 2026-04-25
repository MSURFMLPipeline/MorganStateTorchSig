

def writing_YOLO_Noise_Training_Dataset(self):

        for i, (train_noise_data, train_noise_labels) in tqdm(
            enumerate(self.noise_training_static_dataset),
            total=len(self.noise_training_static_dataset),
            desc="\nWriting YOLO Training Dataset"):

            file_base = str(i).zfill(10)
            train_noise_label_file = f"{self.noise_train_label_dir}/{file_base}.txt"
            train_noise_img_file = f"{self.noise_train_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            train_noise_img = train_noise_data.detach().numpy() if hasattr(train_noise_data, 'detach') else np.array(train_noise_data)
            train_noise_img = np.squeeze(train_noise_img).astype(np.float32)
            train_noise_img = np.abs(train_noise_img)
            train_noise_img = train_noise_img - train_noise_img.min()
            train_noise_img = np.maximum(train_noise_img, 1e-12)
            train_noise_img = np.nan_to_num(train_noise_img, nan=0.0, posinf=0.0, neginf=0.0)

            if train_noise_img.max() > 0:
                train_noise_img = train_noise_img / train_noise_img.max()
                train_noise_img = (train_noise_img * 255).astype(np.uint8)
                train_noise_img = np.stack([train_noise_img, train_noise_img, train_noise_img], axis=-1)
                Image.fromarray(train_noise_img).save(train_noise_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(train_noise_label_file, "w") as f:
                if isinstance(train_noise_labels[0], (list, tuple, np.ndarray)):
                    for l in train_noise_labels:
                        f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")
                else:
                    l = train_noise_labels
                    f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")

print("\nYOLO Training Noise Dataset Generation Complete.")

def writing_YOLO_Noise_Validation_Dataset(self):

        for i, (validation_noise_data, validation_noise_labels) in tqdm(
            enumerate(self.noise_validation_static_dataset),
            total=len(self.noise_validation_static_dataset),
            desc="\nWriting YOLO Validation Dataset"):

            file_base = str(i).zfill(10)
            val_noise_label_file = f"{self.noise_val_label_dir}/{file_base}.txt"
            val_noise_img_file = f"{self.noise_val_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            val_noise_img = validation_noise_data.detach().numpy() if hasattr(validation_noise_data, 'detach') else np.array(validation_noise_data)
            val_noise_img = np.squeeze(val_noise_img).astype(np.float32)
            val_noise_img = np.abs(val_noise_img)
            val_noise_img = val_noise_img - val_noise_img.min()
            val_noise_img = np.maximum(val_noise_img, 1e-12)
            val_noise_img = np.nan_to_num(val_noise_img, nan=0.0, posinf=0.0, neginf=0.0)

            if val_noise_img.max() > 0:
                val_noise_img = val_noise_img / val_noise_img.max()
                val_noise_img = (val_noise_img * 255).astype(np.uint8)
                val_noise_img = np.stack([val_noise_img, val_noise_img, val_noise_img], axis=-1)
                Image.fromarray(val_noise_img).save(val_noise_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(val_noise_label_file, "w") as f:
                if isinstance(validation_noise_labels[0], (list, tuple, np.ndarray)):
                    for l in validation_noise_labels:
                        f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")
                else:
                    l = validation_noise_labels
                    f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")

print("\nYOLO Validation Noise Dataset Generation Complete.")

def writing_YOLO_Noise_Testing_Dataset(self):

        for i, (test_noise_data, test_noise_labels) in tqdm(
            enumerate(self.noise_testing_static_dataset),
            total=len(self.noise_testing_static_dataset),
            desc="\nWriting YOLO Test Dataset"):

            file_base = str(i).zfill(10)
            test_noise_label_file = f"{self.noise_test_label_dir}/{file_base}.txt"
            test_noise_img_file = f"{self.noise_test_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            test_noise_img = test_noise_data.detach().numpy() if hasattr(test_noise_data, 'detach') else np.array(test_noise_data)
            test_noise_img = np.squeeze(test_noise_img).astype(np.float32)
            test_noise_img = np.abs(test_noise_img)
            test_noise_img = test_noise_img - test_noise_img.min()
            test_noise_img = np.maximum(test_noise_img, 1e-12)
            test_noise_img = np.nan_to_num(test_noise_img, nan=0.0, posinf=0.0, neginf=0.0)

            if test_noise_img.max() > 0:
                test_noise_img = test_noise_img / test_noise_img.max()
                test_noise_img = (test_noise_img * 255).astype(np.uint8)
                test_noise_img = np.stack([test_noise_img, test_noise_img, test_noise_img], axis=-1)
                Image.fromarray(test_noise_img).save(test_noise_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
            with open(test_noise_label_file, "w") as f:
                if isinstance(test_noise_labels[0], (list, tuple, np.ndarray)):
                    for l in test_noise_labels:
                        f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")
                else:
                    l = test_noise_labels
                    f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")

print("\nYOLO Testing Noise Dataset Generation Complete.")
    
    
