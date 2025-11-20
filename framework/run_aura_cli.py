#!/usr/bin/env python3
"""AuraGen Framework CLI: Run full workflow prompt/image->video"""
import argparse
import torch
from framework.aura_framework import AuraFramework
from utils.video_output import save_video
from utils.music_output import music_to_wav
from utils.dialogue_output import save_dialogue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraGen Full Framework")
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--reference_images', nargs='*', default=None)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--output_prefix', type=str, default='output/aura_output')
    args = parser.parse_args()
    model_config = {
        'dim': 2048,
        'num_chains': 12,
        'max_frames': 432000,
        'video_resolution': (1920, 1080)
    }
    aura = AuraFramework(model_config)
    results = aura.generate(args.prompt, args.reference_images, num_frames=args.frames)

    save_video(results['video_latent'], args.output_prefix+'.mp4')
    music_to_wav(results['music'], args.output_prefix+'.wav')
    save_dialogue(results['dialogue'], args.output_prefix+'.txt')
    print(f"Outputs saved to prefix {args.output_prefix}")
