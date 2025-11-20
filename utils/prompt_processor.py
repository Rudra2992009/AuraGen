import torch
from PIL import Image
from torchvision import transforms
import numpy as np

class PromptProcessor:
    """Handle textual and image prompts for video generation."""
    
    def __init__(self, text_max_len: int = 512):
        self.text_max_len = text_max_len
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    def process_text(self, text: str) -> torch.Tensor:
        # Dummy tokenizer, replace with actual tokenizer
        tokens = torch.randint(0, 50000, (self.text_max_len,))
        return tokens
    def process_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor
    def process_prompt(self, text: str, images: list = None) -> dict:
        output = {'tokens': self.process_text(text)}
        if images:
            output['image_refs'] = [self.process_image(img) for img in images]
        return output
