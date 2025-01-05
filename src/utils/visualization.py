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