class MSUTorch_YOLO_Model(MSUDataBlock):
    def __init__(self):
        super().__init__()

    def YOLO_Model_training(self):
        print("Starting YOLO training")
        self.model = YOLO("yolov8n.pt")
        self.model.train(
        data="/home/zismi2/Desktop/yolo_Detector.yaml",
        epochs=train["epochs"],
        imgsz=224,
        batch=train["batch_size"],
        )
        self.save_dir = str(self.model.trainer.save_dir)
        print("Training finished")
        
    def YOLO_Model_Noise_training(self):
        print("Starting YOLO training")
        self.noise_model = YOLO("yolov8n.pt")
        self.noise_model.train(
        data="/home/zismi2/Desktop/yolo_Detector_2.yaml",
        epochs=train["epochs"],
        imgsz=224,
        batch=train["batch_size"],
        )
        self.noise_save_dir = str(self.noise_model.trainer.save_dir) 
        print("Training finished")