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
        self.is_loading = False  # New flag to track loading state

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

            # Only raise error if we're not in loading state and thresholds are exceeded
            if not self.is_loading and (memory_percent > self.memory_threshold or gpu_percent > self.gpu_threshold):
                raise RuntimeError(f"Resource limits exceeded: Memory: {memory_percent}%, GPU: {gpu_percent}%")
            
            time.sleep(1)  # Check every second
    def wait_for_memory_clearance(self, timeout=60):
        """Wait for memory to clear below threshold"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            memory_percent = psutil.virtual_memory().percent
            gpu_percent = 0
            if torch.cuda.is_available():
                gpu_devices = GPUtil.getGPUs()
                if gpu_devices:
                    gpu_percent = gpu_devices[0].memoryUtil * 100
            
            if memory_percent < self.memory_threshold and gpu_percent < self.gpu_threshold:
                return True
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            time.sleep(2)
        return False

    def set_loading_state(self, is_loading):
        """Set whether the system is currently loading models"""
        self.is_loading = is_loading

def main():
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # Initialize configuration and resource monitor first
    config_path = Path(__file__).resolve().parent.parent / "config/config.yaml"
    config = ConfigManager(config_path)
    
    resource_monitor = ResourceMonitor(
        memory_threshold=config.get("memory_threshold", 95),
        gpu_threshold=config.get("gpu_threshold", 95)
    )
    
    # Start monitoring and wait for initial memory clearance
    resource_monitor.start_monitoring()
    
    try:
        logger.info("Preparing to load models...")
        resource_monitor.set_loading_state(True)
        
        generator = CharacterGenerator(
            model_path=config.get("model_path"),
            vae_path=config.get("vae_path"),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        controlnet_handler = ControlNetHandler(controlnet_model=config.get("controlnet_model"))
        
        dataset_handler = DatasetHandler(
            data_dir=config.get("data_dir"),
            cache_dir=config.get("cache_dir")
        )

        resource_monitor.set_loading_state(False)
        
        # Generate character with error checking
        logger.info("Starting image generation...")
        try:
            character = generator.generate(
                prompt=config.get("prompt"),
                negative_prompt=config.get("negative_prompt"),
                num_inference_steps=config.get("num_inference_steps", 50),
                guidance_scale=config.get("guidance_scale", 7.5)
            )
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate image: {str(e)}")
        
        if character is None:
            raise RuntimeError("Image generation failed - no image was returned")
        
        # Process with ControlNet if needed
        control_mode = config.get("control_mode")
        if control_mode and character is not None:
            try:
                character = controlnet_handler.process_condition(character, control_mode)
            except Exception as e:
                logger.error(f"ControlNet processing failed: {str(e)}")
                # Continue with unprocessed character if ControlNet fails
        
        # Visualize with error handling
        if character is not None:
            logger.info("Saving generated image...")
            if not visualize_generations(character, save_path=Path("outputs")):
                logger.error("Failed to save visualization")
        else:
            raise RuntimeError("No valid image to visualize")
            
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise
    finally:
        resource_monitor.stop_monitoring()

if __name__ == "__main__":
    main()
