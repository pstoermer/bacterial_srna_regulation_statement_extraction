import yaml

from typing import Dict

class ConfigManager:
    """
    Manages the application configuration, specifically for accessing model paths
    and other configurations defined in a YAML file.

    Attributes:
        config (dict): A dictionary containing the loaded configuration.
    """
    def __init__(self, config_path:str):
        """
        Initializes the ConfigManager with the path to the configuration file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        self.config = self.load_config(config_path)

    @staticmethod
    def load_config(path:str) -> Dict:
        """
        Loads the configuration from a YAML file.

        Args:
            path (str): The path to the YAML file to load.

        Returns:
            dict: The configuration as a dictionary.
        """
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, model_name:str) -> str:
        """
        Retrieves the path for a specified model from the configuration.

        Args:
            model_name (str): The name of the model to retrieve the path for.

        Returns:
            str: The path to the model.
        """
        return self.config['model_paths'][model_name]

