from MSUTorch.data_generation.clean import run_clean_pipeline
from MSUTorch.data_generation.emi import run_emi_pipeline
from MSUTorch.data_generation.clean_plus_emi import run_clean_plus_emi_pipeline
import os 
import yaml
#---------------
#Loading YAMLs
#---------------
_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
ds = config["dataset"]
train = config["training"]
validation = config["validation"]
test = config["test"]
num_signals = config["num_signals"]

class_lists=config["class_list"]
snr=config["snr_db"]
noise=config["noise_power"]
root = ds["root"]
os.makedirs(root,  exist_ok=True)
mode=ds["mode"]

root = os.path.join(ds["root"], mode)
with open(os.path.join(_DIR, "yolo_detector_2.yaml"), "r") as f:
    detector = yaml.safe_load(f)
dt=detector["names"]
paths=detector["paths"]


with open(os.path.join(_DIR, "yolo_detector.yaml"), "r") as f:
    detector = yaml.safe_load(f)
dt=detector["names"]
paths=detector["paths"]
disk_root=paths["root"]

# In order to run the pipeline, whichever pipeline you dont want to run, you can just comment that out
if __name__ == "__main__":
 
    # run_clean_pipeline()
    run_clean_plus_emi_pipeline()
    # run_emi_pipeline()
