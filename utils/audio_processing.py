import torch
import numpy as np
from typing import List, Dict
import librosa

class AudioProcessor:
    """Process and enhance audio output (music, dialogue, sound effects)."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def apply_reverb(self, audio: np.ndarray, room_size: float = 0.5) -> np.ndarray:
        """Apply reverb effect to audio."""
        # Simplified reverb using convolution
        impulse_length = int(self.sample_rate * 0.5)
        impulse = np.random.randn(impulse_length) * np.exp(-np.arange(impulse_length) / (self.sample_rate * room_size))
        return np.convolve(audio, impulse, mode='same')
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio levels."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val * 0.95
        return audio
    
    def mix_audio_tracks(self, tracks: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """Mix multiple audio tracks with optional weights."""
        if weights is None:
            weights = [1.0] * len(tracks)
        
        max_length = max(len(t) for t in tracks)
        mixed = np.zeros(max_length)
        
        for track, weight in zip(tracks, weights):
            padded = np.pad(track, (0, max_length - len(track)))
            mixed += padded * weight
        
        return self.normalize_audio(mixed)
    
    def extract_tempo(self, audio: np.ndarray) -> float:
        """Extract tempo from audio."""
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        return tempo
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Time-stretch audio without changing pitch."""
        return librosa.effects.time_stretch(audio, rate=rate)
