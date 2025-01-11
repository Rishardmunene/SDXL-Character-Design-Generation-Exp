from typing import Optional
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Adjusted sys.path
import torch
from models.character_generator import CharacterGenerator
from models.controlnet_handler import ControlNetHandler
from data.dataset_handler import DatasetHandler
from utils.config_manager import ConfigManager  
from utils.visualization import visualize_generations
from utils.logger import setup_logger
import psutil
import GPUtil
from threading import Thread
import time

class ResourceMonitor:
    def __init__(self, memory_threshold=90, gpu_threshold=90):
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.should_stop = False
        self.monitoring_thread = None

    def start_monitoring(self):
        self.monitoring_thread = Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        self.should_stop = True
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitor_resources(self):
        while not self.should_stop:
            # Check CPU memory usage
            memory_percent = psutil.virtual_memory().percent
            
            # Check GPU usage if available
            gpu_percent = 0
            if torch.cuda.is_available():
                gpu_devices = GPUtil.getGPUs()
                if gpu_devices:
                    gpu_percent = gpu_devices[0].memoryUtil * 100

            if memory_percent > self.memory_threshold or gpu_percent > self.gpu_threshold:
                raise RuntimeError(f"Resource limits exceeded: Memory: {memory_percent}%, GPU: {gpu_percent}%")
            
            time.sleep(1)  # Check every second

def main():
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config_path = Path(__file__).resolve().parent.parent / "config/config.yaml"
    config = ConfigManager(config_path) 
    
    # Initialize model
    generator = CharacterGenerator(
        model_path=config.get("model_path"),
        vae_path=config.get("vae_path"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Initialize ControlNet handler
    controlnet_handler = ControlNetHandler(controlnet_model=config.get("controlnet_model"))
    
    # Initialize dataset handler
    dataset_handler = DatasetHandler(
        data_dir=config.get("data_dir"),
        cache_dir=config.get("cache_dir")
    )
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(
        memory_threshold=config.get("memory_threshold", 90),
        gpu_threshold=config.get("gpu_threshold", 90)
    )
    resource_monitor.start_monitoring()
    
    try:
        # Generate character
        character = generator.generate(
            prompt=config.get("prompt"),
            negative_prompt=config.get("negative_prompt"),
            num_inference_steps=config.get("num_inference_steps", 50),
            guidance_scale=config.get("guidance_scale", 7.5)
        )
        
        # Process with ControlNet if needed
        control_mode = config.get("control_mode")
        if control_mode:
            character = controlnet_handler.process_condition(character, control_mode)
        
        # Visualize and save results
        visualize_generations(character, save_path=Path("outputs"))
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise

    finally:
        resource_monitor.stop_monitoring()

if __name__ == "__main__":
    main()
