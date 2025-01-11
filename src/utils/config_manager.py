import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Initialize logger
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: Path):
        logger.info(f"Loading configuration from {config_path}")
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.config.get(key, default)
        
    def update(self, key: str, value: Any):
        self.config[key] = value
        self._save_config()
        
    def _save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)