from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
from typing import Optional, List, Union
import numpy as np

class CharacterGenerator:
    def __init__(self, model_path: str, vae_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Initialize the pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            variant="fp16" if torch.cuda.is_available() else None,
            use_safetensors=True
        )
        
        # Load custom VAE if provided
        if vae_path:
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                torch_dtype=self.dtype
            )
            self.pipeline.vae = vae
        
        # Move to device and enable memory optimization
        self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing()
        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
    
    def generate(self, prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate an image based on the given prompt.
        
        Args:
            prompt (str): The input prompt for image generation
            negative_prompt (str, optional): The negative prompt
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): Guidance scale for generation
            
        Returns:
            PIL.Image: The generated image
        """
        # Ensure consistent dtype across the pipeline
        self.pipeline.to(dtype=self.dtype)
        
        # Generate the image
        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
        
        return images[0]  # Return the first generated image