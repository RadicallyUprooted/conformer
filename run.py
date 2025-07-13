import hydra
from omegaconf import DictConfig
import subprocess
import sys

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    This script acts as a dispatcher, running either the training or 
    inference script based on the provided configuration. It forwards
    all command-line overrides to the appropriate script.
    """
    
    overrides = sys.argv[1:]

    if cfg.inference.audio_path:
        script_to_run = "inference.py"
        print(f"--- Running Inference ---")
    else:
        script_to_run = "train.py"
        print(f"--- Running Training ---")

    command = ["python", script_to_run] + overrides
    print(f"Executing command: {' '.join(command)}\n")
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing script: {script_to_run}")
        print(f"Return code: {e.returncode}")
    except FileNotFoundError:
        print(f"\nError: Could not find the script '{script_to_run}'. Make sure it is in the correct directory.")

if __name__ == "__main__":
    main()