import torch
import numpy as np
from typing import List, Tuple
import cv2

class VideoFrameProcessor:
    """Advanced video frame processing and effects."""
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.resolution = resolution
    
    def apply_color_grading(self, frame: np.ndarray, style: str = 'cinematic') -> np.ndarray:
        """Apply color grading to frame."""
        if style == 'cinematic':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            frame[:,:,1] = frame[:,:,1] * 0.8  # Reduce saturation
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        elif style == 'warm':
            frame[:,:,0] = np.clip(frame[:,:,0] * 1.1, 0, 255)
            frame[:,:,2] = np.clip(frame[:,:,2] * 0.9, 0, 255)
        return frame
    
    def add_motion_blur(self, frame: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Add motion blur effect."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(frame, -1, kernel)
    
    def add_fade_transition(self, frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """Blend two frames with fade transition."""
        return cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
    
    def add_text_overlay(self, frame: np.ndarray, text: str, position: Tuple[int, int] = (50, 50)) -> np.ndarray:
        """Add text overlay to frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, position, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
