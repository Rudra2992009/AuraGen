#!/usr/bin/env python3
"""CLI for long term video imagination generation"""
import argparse
import torch
from framework.aura_longterm import AuraLongTerm
from utils.video_output import save_video
from utils.music_output import music_to_wav
from utils.dialogue_output import save_dialogue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraGen Hyper-Realistic Long Video Generator")
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--reference_images', nargs='*', default=None)
    parser.add_argument('--duration', type=int, default=7200)
    parser.add_argument('--output_prefix', type=str, default='output/auragen_longoutput')
    args = parser.parse_args()
    model_config = {
        'dim': 2048,
        'num_chains': 12,
        'max_frames': 432000,
        'video_resolution': (1920, 1080)
    }
    aura_longterm = AuraLongTerm(model_config)
    results = aura_longterm.generate(args.prompt, args.reference_images, duration=args.duration)
    save_video(results['video_latent'], args.output_prefix+'.mp4')
    music_to_wav(results['music'], args.output_prefix+'.wav')
    save_dialogue(results['dialogue'], args.output_prefix+'.txt')
    print(f"Hyper-real long video outputs saved with prefix {args.output_prefix}")
