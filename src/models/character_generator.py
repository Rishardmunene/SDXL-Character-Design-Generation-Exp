from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
from typing import Optional, List, Union
import numpy as np

class CharacterGenerator:
    def __init__(
        self,
        model_path: str,
        vae_path: Optional[str],
        device: torch.device,
        enable_memory_optimization: bool = True
    ):
        self.device = device
        self.pipeline = self._initialize_pipeline(model_path, vae_path)
        if enable_memory_optimization:
            self._optimize_memory_usage()
    
    def _initialize_pipeline(self, model_path, vae_path):
    # Load the pipeline with float32 precision initially
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            vae=AutoencoderKL.from_pretrained(vae_path),
            torch_dtype=torch.float32,  # Set to float32 explicitly
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        pipeline = pipeline.to(self.device)
        
        # Optional: Enable memory efficient attention
        pipeline.enable_attention_slicing()
        
        return pipeline

def generate(self, prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5):
    # Ensure we're using float32 for the VAE
    self.pipeline.vae.to(dtype=torch.float32)
    
    # Generate the image
    images = self.pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images
    
    return images[0]  # Return the first generated image