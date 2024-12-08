import yaml

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config