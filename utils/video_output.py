import torch
import numpy as np
import imageio
from pathlib import Path

def latent_to_video_frames(video_latent, res=(1920, 1080)):
    """Convert model latent output to video frames (as numpy arrays)."""
    batch, seq, _ = video_latent.shape
    pixels = video_latent.view(batch, seq, 3, res[0]//8, res[1]//8)
    # Upsampling (dummy - replace with actual upsampler)
    frames = []
    for i in range(seq):
        frame = pixels[0,i].permute(1,2,0).detach().cpu().numpy()
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames

def save_video(frames, filename, fps=24):
    """Save frame list as MP4 video."""
    imageio.mimsave(filename, frames, fps=fps)

if __name__ == "__main__":
    video_latent = torch.randn(1,120,3*(1920//8)*(1080//8))
    frames = latent_to_video_frames(video_latent)
    Path('output').mkdir(exist_ok=True)
    out_path = 'output/dummy_video.mp4'
    save_video(frames, out_path, fps=24)
    print(f"Video saved: {out_path}")
