import torch
import cv2
import numpy as np
from pathlib import Path

class ReferenceImageValidator:
    """Validates reference images for safety (no nudity/explicit content)."""
    def __init__(self):
        # Placeholder for actual nudity detection model
        # In production, use pre-trained NSFW detection or similar
        pass
    
    def is_safe_image(self, image_path: str) -> bool:
        """Check if image is safe (no nudity/explicit content)."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Placeholder heuristic - in production use proper NSFW detector
            # For now, return True (implement actual detection)
            # Example: integrate with NudeNet or similar
            return True
        except Exception as e:
            print(f"Error validating image: {e}")
            return False

class ImagePreprocessor:
    """Preprocess and prepare reference images for model input."""
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_and_preprocess(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image to tensor."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img
