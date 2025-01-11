import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import Union, List
import logging

logger = logging.getLogger(__name__)

def visualize_generations(images: Union[Image.Image, List[Image.Image]], save_path: Path):
    """
    Visualize and save generated images.
    
    Args:
        images: Single PIL Image or list of PIL Images
        save_path: Directory to save the visualization
    """
    # Ensure save_path exists
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Convert single image to list for consistent handling
    if isinstance(images, Image.Image):
        images = [images]
    
    if not images:
        logger.error("No images provided for visualization")
        return
    
    # Create figure and axes
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
    
    # Convert single axis to list for consistent handling
    if len(images) == 1:
        axes = [axes]
    
    # Plot each image
    for idx, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Generation {idx+1}')
    
    # Save the figure
    plt.tight_layout()
    save_file = save_path / "generated_characters.png"
    plt.savefig(save_file)
    plt.close()
    
    # Also save individual images
    for idx, img in enumerate(images):
        img_save_path = save_path / f"character_{idx+1}.png"
        img.save(img_save_path)
    
    logger.info(f"Saved visualizations to {save_path}")