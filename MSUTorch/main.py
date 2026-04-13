from data.dataset import MSUDataBlock
from models.yolo_model import MSUTorch_YOLO_Model
from models.eval import MSUTorch_Eval

#--------------------
#EMI (Completed) 
#-------------------
data=MSUDataBlock()
data.Dataset_Mode()
data.selection_conditions()
data.metadata_tone()
data.AWGN_noise_power()
data.signal_impairments()
data.signal_training_impaired_dataset()
data.signal_testing_impaired_dataset()
data.signal_validation_impaired_dataset()
data.creating_noisy_validation_dataloader()
data.creating_noisy_testing_dataloader()
data.creating_noisy_training_dataloader()
data.writing_noisy_testing_dataset()
data.writing_noisy_validation_dataset()
data.writing_noisy_training_dataset()
data.reading_noisy_validation_dataset()
data.reading_noisy_testing_dataset()
data.reading_noisy_training_dataset()
data.inspecting_noisy_training_batch()
data.inspecting_noisy_testing_batch()
data.inspecting_noisy_validation_batch()
data.noisy_spectrogram_directories()
data.writing_YOLO_Noise_Training_Dataset()
data.writing_YOLO_Noise_Validation_Dataset()
data.writing_YOLO_Noise_Testing_Dataset()
ML=MSUTorch_YOLO_Model()
ML.YOLO_Model_Noise_training()
ML.YOLO_Noise_Model_validation()
ML.export_YOLO_Noise_model()

#--------------------
#Clean+EMI
#--------------------




#--------------------
#Clean Data Generation Pipeline (Completed)
#--------------------
"""
data=MSUDataBlock()
data.Dataset_Mode()
data.selection_conditions()
data.update_return_metadata()
data.training_dataset()
data.testing_dataset()
data.validation_dataset()
data.creating_testing_dataloader()
data.creating_training_dataloader()
data.creating_validation_dataloader()
data.writing_testing_dataset()
data.writing_training_dataset()
data.writing_validation_dataset()
data.reading_training_dataset()
data.reading_validation_dataset()
data.reading_testing_dataset()
data.inspecting_training_batch()
data.inspecting_test_batch()
data.inspecting_validation_batch()


data.spectrogram_image_directories()
data.writing_YOLO_Validation_Dataset()
data.writing_YOLO_Testing_Dataset()
data.writing_YOLO_Training_Dataset()
ML=MSUTorch_YOLO_Model()
ML.YOLO_Model_training()
ML.YOLO_Model_validation()
ML.export_YOLO_model()
ML.performance()
EV=MSUTorch_Eval()
EV.load_model()
EV.training_model_performance()
EV.testing_model_performance()"""
