# models/character_generator.py
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
from typing import Optional, List, Union, Dict
from pathlib import Path
import numpy as np

class CharacterGenerator:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        vae: Optional[AutoencoderKL] = None,
        enable_memory_optimization: bool = True
    ):
        self.device = device
        self.pipeline = self._initialize_pipeline(model_path, vae)
        if enable_memory_optimization:
            self._optimize_memory_usage()
    
    def _initialize_pipeline(self, model_path: str, vae: Optional[AutoencoderKL]) -> StableDiffusionXLPipeline:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device.type == "cuda" else None
        )
        if vae:
            pipeline.vae = vae
        pipeline.to(self.device)
        return pipeline
    
    def _optimize_memory_usage(self):
        self.pipeline.enable_attention_slicing()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> List[np.ndarray]:
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            
        with torch.inference_mode():
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
                callback=callback
            ).images
            
        return images