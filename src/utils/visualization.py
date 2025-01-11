import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import Union, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_single_image(image: Image.Image, save_path: Path, index: int = 1) -> bool:
    """Helper function to save a single image"""
    try:
        if not isinstance(image, Image.Image):
            logger.error(f"Invalid image type: {type(image)}")
            return False
            
        img_save_path = save_path / f"character_{index}.png"
        image.save(img_save_path)
        return True
    except Exception as e:
        logger.error(f"Failed to save image {index}: {str(e)}")
        return False

def visualize_generations(image_input: Optional[Union[Image.Image, List[Image.Image]]], save_path: Path) -> bool:
    """
    Visualize and save generated images.
    
    Args:
        image_input: Single PIL Image or list of PIL Images
        save_path: Directory to save the visualization
    
    Returns:
        bool: True if visualization was successful, False otherwise
    """
    try:
        # Early return if no valid input
        if image_input is None:
            logger.error("No image provided for visualization")
            return False

        # Convert to list if single image
        images = [image_input] if isinstance(image_input, Image.Image) else image_input

        # Validate images
        if not images or not any(isinstance(img, Image.Image) for img in images):
            logger.error("No valid images found for visualization")
            return False

        # Create output directory
        save_path.mkdir(parents=True, exist_ok=True)

        # Save individual images first
        valid_images = []
        for idx, img in enumerate(images, 1):
            if isinstance(img, Image.Image):
                if save_single_image(img, save_path, idx):
                    valid_images.append(img)

        if not valid_images:
            logger.error("No valid images could be saved")
            return False

        # Create visualization grid
        fig = plt.figure(figsize=(5*len(valid_images), 5))
        
        for idx, img in enumerate(valid_images, 1):
            ax = fig.add_subplot(1, len(valid_images), idx)
            ax.imshow(np.array(img))
            ax.axis('off')
            ax.set_title(f'Generation {idx}')

        # Save combined visualization
        plt.tight_layout()
        plt.savefig(save_path / "generated_characters.png")
        plt.close()

        logger.info(f"Successfully saved {len(valid_images)} images to {save_path}")
        return True

    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        plt.close()  # Ensure figure is closed even if error occurs
        return False