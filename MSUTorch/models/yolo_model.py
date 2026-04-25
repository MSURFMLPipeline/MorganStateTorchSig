class MSUTorch_YOLO_Model(MSUDataBlock):
    def __init__(self):
        super().__init__()

     def YOLO_Model_training(self):
        print("Starting YOLO training")
        self.model = YOLO("yolov8n.pt")
        self.model.train(
        data="/home/zismi2/Desktop/yolo_Detector.yaml",
        epochs=train["epochs"],
        patience=20,
        imgsz=512,
        batch=train["batch_size"],
        rect=False,
        lr0=0.001,
        cache=False,
        )
        self.save_dir = str(self.model.trainer.save_dir)
        print("Training finished \n")
        best_path=f"{self.save_dir}/weights/best.pt"
        print(f"Here are the best model weights: {best_path}")
        self.model.info()
        print(self.model.names)
        
    def YOLO_Model_emi_training(self):
        print("Starting YOLO training")
        self.emi_model = YOLO("yolov8n.pt")
        self.emi_model.train(
        data="/home/zismi2/Desktop/yolo_Detector_2.yaml",
        epochs=train["epochs"],
        imgsz=512,
        rect=False,
        augment=False,
        patience=20,
        batch=train["batch_size"],
        lr0=0.001,
        cache=False,
        )
        self.emi_save_dir = str(self.emi_model.trainer.save_dir) 
        print("Training finished")
        best_path=f"{self.emi_save_dir}/weights/best.pt"
        print(f"Here are the best model weights: {best_path}")
        self.emi_model.info()
        print(self.emi_model.names)
        
    def YOLO_Model_Clean_EMI_training(self):
        print("Starting YOLO training")
        self.clean_emi_model = YOLO("yolov8n.pt")
        self.clean_emi_model.train(
        data="/home/zismi2/Desktop/yolo_Detector_3.yaml",
        epochs=train["epochs"],
        imgsz=512,
        patience=20,
        batch=train["batch_size"],
        rect=False,
        augment=False,
        lr0=0.001,
        cache=False,
        )
        self.clean_emi_save_dir = str(self.clean_emi_model.trainer.save_dir) 
        print("Training finished")
        best_path=f"{self.clean_emi_save_dir}/weights/best.pt"
        print(f"Here are the best model weights: {best_path}")
        self.clean_emi_model.info()
        print(self.clean_emi_model.names)

