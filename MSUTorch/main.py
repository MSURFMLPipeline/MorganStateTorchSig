from MSUTorch.data_generation.clean import run_clean_pipeline
from MSUTorch.data_generation.emi import run_emi_pipeline
from MSUTorch.data_generation.run_clean_plus_emi import run_clean_plus_emi_pipeline

# Whichever one you dont want to run you can comment that out
if __name__ == "__main__":
 
    # run_clean_pipeline()
    run_clean_plus_emi_pipeline()
    # run_emi_pipeline()
