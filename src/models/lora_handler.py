class LoRAHandler:
    def __init__(self, base_model):
        self.base_model = base_model
        
    def load_lora(self, lora_path: str, alpha: float = 0.75):
        self.base_model.load_lora_weights(lora_path)
        self.base_model.fuse_lora(alpha)
        
    def unload_lora(self):
        self.base_model.unload_lora_weights()