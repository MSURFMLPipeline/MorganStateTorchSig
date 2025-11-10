Saving to YAML
from MSUTorch.configs.dummy_config import DatasetMetadata
from MSUTorch.storage.yaml import save_dataset_yaml, load_dataset_yaml
filepath = "./datasets/yaml_test_dataset"

Loading from YAML
dataset = default_dataset(seed=42, target_labels=["class_name","snr_db"], impairment_level=None) # basic default dataset used for testing
save_dataset_yaml(filepath, dataset)
print(next(dataset))
print(next(dataset))
print(next(dataset))
Loading from YAML
Now we load a copy of the same dataset. Because the copy loads the same random seed from YAML, the values returned below should match the values above.

dataset_copy = load_dataset_yaml(filepath)
print(next(dataset_copy))
print(next(dataset_copy))
print(next(dataset_copy))
