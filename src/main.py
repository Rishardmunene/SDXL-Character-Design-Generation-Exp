from typing import Optional, Dict, Any
import logging
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from pathlib import Path
import torch
from models.character_generator import CharacterGenerator
from models.controlnet_handler import ControlNetHandler
from data.dataset_handler import DatasetHandler
from utils.config_manager import ConfigManager  
from utils.visualization import visualize_generations
from utils.logger import setup_logger

def main():
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config = ConfigManager("config/config.yaml") 
    
    # Initialize model
    generator = CharacterGenerator(
        model_path=config.get("model_path"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Initialize ControlNet handler
    controlnet_handler = ControlNetHandler(controlnet_model=config.get("controlnet_model"))
    
    # Initialize dataset handler
    dataset_handler = DatasetHandler(
        data_dir=config.get("data_dir"),
        cache_dir=config.get("cache_dir")
    )
    
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

if __name__ == "__main__":
    main()