 
class MSUTorch_YOLO_Model(MSUDataBlock):
    def __init__(self):
        super().__init__()

    def export_YOLO_Clean_EMI_model(self):
        YOLO_export = self.clean_emi_model.export(format='onnx', opset=18)  # export the model to ONNX format    
    
    def clean_emi_performance(self):
        label_path= os.path.join(self.clean_emi_model.trainer.save_dir, "val_batch1_labels.jpg")
        pred_path = os.path.join(self.clean_emi_model.trainer.save_dir, "val_batch1_pred.jpg")
        
        label = cv2.imread(label_path)
        pred  = cv2.imread(pred_path)
        
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        pred  = cv2.cvtColor(pred,  cv2.COLOR_BGR2RGB)

        f, ax = plt.subplots(1, 2, figsize=(15, 9))
        ax[0].imshow(label)
        ax[0].set_title("Label")
        ax[1].imshow(pred)
        ax[1].set_title("Prediction")
        plt.tight_layout()
        plt.show()
    
    def clean_training_model_performance(self):
        self.clean_model_train = summary(self.clean_emi_model)


    def testing_model_performance(self):
        self.model_test = summary(self.model.test)

    def export_YOLO_Clean_EMI_model(self):
        YOLO_export = self.clean_emi_model.export(format='onnx', opset=18)  # export the model to ONNX format    
    
    def clean_emi_performance(self):
        label_path= os.path.join(self.clean_emi_model.trainer.save_dir, "val_batch1_labels.jpg")
        pred_path = os.path.join(self.clean_emi_model.trainer.save_dir, "val_batch1_pred.jpg")
        
        label = cv2.imread(label_path)
        pred  = cv2.imread(pred_path)
        
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        pred  = cv2.cvtColor(pred,  cv2.COLOR_BGR2RGB)

        f, ax = plt.subplots(1, 2, figsize=(15, 9))
        ax[0].imshow(label)
        ax[0].set_title("Label")
        ax[1].imshow(pred)
        ax[1].set_title("Prediction")
        plt.tight_layout()
        plt.show()
    
    def clean_training_model_performance(self):
        self.clean_model_train = summary(self.clean_emi_model)


    def testing_model_performance(self):
        self.model_test = summary(self.model.test)

    def performance(self):
        label_path= os.path.join(self.model.trainer.save_dir, "val_batch1_labels.jpg")
        pred_path = os.path.join(self.model.trainer.save_dir, "val_batch1_pred.jpg")
        
        
        label = cv2.imread(label_path)
        pred  = cv2.imread(pred_path)
        
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        pred  = cv2.cvtColor(pred,  cv2.COLOR_BGR2RGB)

        f, ax = plt.subplots(1, 2, figsize=(15, 9))
        ax[0].imshow(label)
        ax[0].set_title("Label")
        ax[1].imshow(pred)
        ax[1].set_title("Prediction")
        plt.tight_layout()
        plt.show()
        

    def training_model_performance(self):
        self.model_train = summary(self.model)