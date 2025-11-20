import torch
import numpy as np
import soundfile as sf

def music_to_wav(music_latent, filename, sample_rate=44100):
    """Convert music tensor output to WAV and save."""
    waveform = music_latent.squeeze().detach().cpu().numpy()
    sf.write(filename, waveform, sample_rate)

if __name__ == "__main__":
    music_latent = torch.randn(120, 512)
    music_to_wav(music_latent, 'output/dummy_music.wav')
    print("Music saved: output/dummy_music.wav")
