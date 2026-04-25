class MSUDataBlock: #Initializing metadata
    def __init__ (self):
        pass

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

    def writing_YOLO_emi_Training_Dataset(self):

        for i, (train_emi_data, train_emi_labels) in tqdm(
            enumerate(self.emi_training_static_dataset),
            total=len(self.emi_training_static_dataset),
            desc="\nWriting YOLO Training Dataset"):

            file_base = str(i).zfill(10)
            train_emi_label_file = f"{self.emi_train_label_dir}/{file_base}.txt"
            train_emi_img_file = f"{self.emi_train_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            train_emi_img = train_emi_data.detach().numpy() if hasattr(train_emi_data, 'detach') else np.array(train_emi_data)
            train_emi_img = np.squeeze(train_emi_img).astype(np.float32)
            train_emi_img = np.abs(train_emi_img)
            train_emi_img = train_emi_img - train_emi_img.min()
            train_emi_img = np.maximum(train_emi_img, 1e-12)
            train_emi_img = np.nan_to_num(train_emi_img, nan=0.0, posinf=0.0, neginf=0.0)

            if train_emi_img.max() > 0:
                train_emi_img = train_emi_img / train_emi_img.max()
                train_emi_img = (train_emi_img * 255).astype(np.uint8)
                train_emi_img = np.stack([train_emi_img, train_emi_img, train_emi_img], axis=-1)
                Image.fromarray(train_emi_img).save(train_emi_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
        with open(train_emi_label_file, "w") as f:
            if len(train_emi_labels) > 0 and isinstance(train_emi_labels[0], (list, tuple, np.ndarray)):
                for l in train_emi_labels:
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
            elif len(train_emi_labels) > 0:
                l = train_emi_labels
                f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

    print("\nYOLO Training EMI Dataset Generation Complete.")

    def writing_YOLO_emi_Validation_Dataset(self):

        for i, (validation_emi_data, validation_emi_labels) in tqdm(
            enumerate(self.emi_validation_static_dataset),
            total=len(self.emi_validation_static_dataset),
            desc="\nWriting YOLO Validation Dataset"):

            file_base = str(i).zfill(10)
            val_emi_label_file = f"{self.emi_val_label_dir}/{file_base}.txt"
            val_emi_img_file = f"{self.emi_val_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            val_emi_img = validation_emi_data.detach().numpy() if hasattr(validation_emi_data, 'detach') else np.array(validation_emi_data)
            val_emi_img = np.squeeze(val_emi_img).astype(np.float32)
            val_emi_img = np.abs(val_emi_img)
            val_emi_img = val_emi_img - val_emi_img.min()
            val_emi_img = np.maximum(val_emi_img, 1e-12)
            val_emi_img = np.nan_to_num(val_emi_img, nan=0.0, posinf=0.0, neginf=0.0)

            if val_emi_img.max() > 0:
                val_emi_img = val_emi_img / val_emi_img.max()
                val_emi_img = (val_emi_img * 255).astype(np.uint8)
                val_emi_img = np.stack([val_emi_img, val_emi_img, val_emi_img], axis=-1)
                Image.fromarray(val_emi_img).save(val_emi_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
        with open(val_emi_label_file, "w") as f:
            if len(validation_emi_labels) > 0 and isinstance(validation_emi_labels[0], (list, tuple, np.ndarray)):
                for l in validation_emi_labels:
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
            elif len(validation_emi_labels) > 0:
                l = validation_emi_labels
                f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n") 
                
    print("\nYOLO Validation EMI Dataset Generation Complete.")
    
    def writing_YOLO_emi_Testing_Dataset(self):

        for i, (test_emi_data, test_emi_labels) in tqdm(
            enumerate(self.emi_testing_static_dataset),
            total=len(self.emi_testing_static_dataset),
            desc="\nWriting YOLO Test Dataset"):

            file_base = str(i).zfill(10)
            test_emi_label_file = f"{self.emi_test_label_dir}/{file_base}.txt"
            test_emi_img_file = f"{self.emi_test_image_dir}/{file_base}.png"

        # -------------------------
        # Convert spectrogram to image
        # -------------------------
            test_emi_img = test_emi_data.detach().numpy() if hasattr(test_emi_data, 'detach') else np.array(test_emi_data)
            test_emi_img = np.squeeze(test_emi_img).astype(np.float32)
            test_emi_img = np.abs(test_emi_img)
            test_emi_img = test_emi_img - test_emi_img.min()
            test_emi_img = np.maximum(test_emi_img, 1e-12)
            test_emi_img = np.nan_to_num(test_emi_img, nan=0.0, posinf=0.0, neginf=0.0)

            if test_emi_img.max() > 0:
                test_emi_img = test_emi_img / test_emi_img.max()
                test_emi_img = (test_emi_img * 255).astype(np.uint8)
                test_emi_img = np.stack([test_emi_img, test_emi_img, test_emi_img], axis=-1)
                Image.fromarray(test_emi_img).save(test_emi_img_file)

        # -------------------------
        # Write YOLO label
        # -------------------------
        with open(test_emi_label_file, "w") as f:
            if len(test_emi_labels) > 0 and isinstance(test_emi_labels[0], (list, tuple, np.ndarray)):
                for l in test_emi_labels:
                    f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
            elif len(test_emi_labels) > 0:
                l = test_emi_labels
                f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

    print("\nYOLO Testing EMI Dataset Generation Complete.")

