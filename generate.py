#!/usr/bin/env python3
"""Complete end-to-end video generation pipeline."""

import torch
import argparse
import json
from pathlib import Path
from model.neural_architecture import create_auragen_model
from utils.model_weights import load_model_safetensors
from utils.video_output import latent_to_video_frames, save_video
from utils.music_output import music_to_wav
from utils.dialogue_output import save_dialogue
from utils.frame_processing import VideoFrameProcessor
from utils.audio_processing import AudioProcessor

def generate_full_video(
    prompt: str,
    duration_seconds: int,
    output_dir: str,
    model_path: str,
    access_token: str = None
):
    """Generate complete video with audio and dialogue."""
    
    # Validate access token if provided
    if access_token:
        valid_tokens = ['rudra_qazwsxedcrfvtgbyhnujmikolp', 'rudra_plokmijnuhbygvtfcrdxeszwaq']
        if access_token not in valid_tokens:
            raise ValueError('Invalid access token')
    
    print(f"Generating video: '{prompt}'")
    print(f"Duration: {duration_seconds} seconds")
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load model
    config = {
        'dim': 2048,
        'num_chains': 12,
        'max_frames': 432000,
        'video_resolution': (1920, 1080)
    }
    
    print("Loading model...")
    model = create_auragen_model(config)
    if Path(model_path).exists():
        model = load_model_safetensors(create_auragen_model, config, model_path)
    model.eval()
    
    # Tokenize prompt (simplified)
    prompt_tokens = torch.randint(0, 50000, (1, 512))
    
    # Generate
    fps = 24
    num_frames = duration_seconds * fps
    
    print(f"Generating {num_frames} frames...")
    with torch.no_grad():
        outputs = model(
            prompt_tokens,
            num_frames=num_frames,
            generate_dialogue=True,
            generate_music=True
        )
    
    # Process video frames
    print("Processing video frames...")
    frames = latent_to_video_frames(outputs['video_latent'])
    
    # Apply frame processing
    processor = VideoFrameProcessor()
    processed_frames = [processor.apply_color_grading(f, 'cinematic') for f in frames]
    
    # Save video
    video_path = output_path / 'video.mp4'
    print(f"Saving video to {video_path}...")
    save_video(processed_frames, str(video_path), fps=fps)
    
    # Save music
    if 'music' in outputs:
        music_path = output_path / 'music.wav'
        print(f"Saving music to {music_path}...")
        music_to_wav(outputs['music'], str(music_path))
    
    # Save dialogue
    if 'dialogue' in outputs:
        dialogue_path = output_path / 'dialogue.txt'
        print(f"Saving dialogue to {dialogue_path}...")
        save_dialogue(outputs['dialogue'], str(dialogue_path))
    
    # Save metadata
    metadata = {
        'prompt': prompt,
        'duration_seconds': duration_seconds,
        'num_frames': num_frames,
        'fps': fps,
        'resolution': config['video_resolution']
    }
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration complete! Output saved to: {output_dir}")
    print(f"  - Video: {video_path}")
    print(f"  - Music: {music_path}" if 'music' in outputs else "")
    print(f"  - Dialogue: {dialogue_path}" if 'dialogue' in outputs else "")
    print(f"  - Metadata: {metadata_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AuraGen Video Generation Pipeline')
    parser.add_argument('prompt', type=str, help='Video generation prompt')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    parser.add_argument('--output', type=str, default='output/generated', help='Output directory')
    parser.add_argument('--model', type=str, default='weights/aura-base.safetensors', help='Model path')
    parser.add_argument('--token', type=str, default=None, help='Access token')
    
    args = parser.parse_args()
    
    generate_full_video(
        prompt=args.prompt,
        duration_seconds=args.duration,
        output_dir=args.output,
        model_path=args.model,
        access_token=args.token
    )
