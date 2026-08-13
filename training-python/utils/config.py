import json
import os

class ConfigManager:
    
    __instance = None
    
    def __init__(self):
        self.config = {}
        
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        if cls.__instance is None:
            cls.__instance = ConfigManager()
        return cls.__instance
        
    @classmethod
    def load_config(cls, file_path: str):
        instance = cls.get_instance()
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                instance.config = json.load(f)
        else:
            raise FileNotFoundError(f"Configuration file {file_path} not found.")
        
    @classmethod
    def get(cls, key, default=None):
        instance = cls.get_instance()
        return instance.config.get(key, default)
        
    @classmethod
    def get_config(cls) -> dict:
        instance = cls.get_instance()
        return instance.config
    
    @classmethod
    def show_config(cls) -> dict:
        instance = cls.get_instance()
        print(json.dumps(instance.config, indent=4))
        return instance.config
    
    @classmethod
    def new_config(cls, config) -> dict:
        instance = cls.get_instance()
        instance.config = config
        return instance.config