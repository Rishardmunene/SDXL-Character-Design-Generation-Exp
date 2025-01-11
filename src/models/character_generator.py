from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
from typing import Optional, List, Union
import numpy as np
from PIL import Image
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class CharacterGenerator:
    def __init__(self, model_path: str, vae_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Initializing CharacterGenerator on {self.device} with {self.dtype}")
        
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
        """
        try:
            logger.info(f"Generating image with prompt: {prompt}")
            
            # Validate inputs
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            
            # Generate the image
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            
            # Validate the output
            if not hasattr(result, 'images') or not result.images:
                raise RuntimeError("Pipeline did not return any images")
            
            generated_image = result.images[0]
            
            # Verify the image
            if not isinstance(generated_image, Image.Image):
                raise TypeError(f"Expected PIL.Image but got {type(generated_image)}")
            
            logger.info("Image generation successful")
            return generated_image
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate image: {str(e)}")