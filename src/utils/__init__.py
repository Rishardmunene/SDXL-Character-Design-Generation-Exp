# utils/__init__.py
from .config_manager import ConfigManager
from .visualization import visualize_generations
from .logger import setup_logger

__all__ = ["ConfigManager", "visualize_generations", "setup_logger"]

# utils/config_manager.py
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
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

# utils/visualization.py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union
import numpy as np

def visualize_generations(
    images: Union[np.ndarray, List[np.ndarray]],
    save_path: Optional[Path] = None,
    show: bool = True
):
    if isinstance(images, np.ndarray):
        images = [images]
        
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
    if len(images) == 1:
        axes = [axes]
        
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "generations.png")
        
    if show:
        plt.show()
    plt.close()

# utils/logger.py
import logging
from pathlib import Path

def setup_logger(
    log_file: Optional[str] = "logs/generation.log",
    level: int = logging.INFO
):
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )