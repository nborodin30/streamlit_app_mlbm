import yaml

def load_config(config_path="conf/config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg