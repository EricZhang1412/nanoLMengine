import os
import yaml

def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file {config_path}: {exc}")