#!/usr/bin/env python3
"""CLI for full creative suite: hyper-real video with style and adaptive music"""
import argparse
import torch
from framework.full_creative_suite import AuraFullCreativeSuite
from utils.video_output import save_video
from utils.music_output import music_to_wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraGen Full Creative Suite Generator")
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--styles', nargs='*', type=int, default=[0])
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--output_prefix', type=str, default='output/auragen_creativeoutput')
    args = parser.parse_args()
    model_config = {
        'dim': 2048,
        'num_chains': 12,
        'max_frames': 432000,
        'video_resolution': (1920, 1080)
    }
    full_suite = AuraFullCreativeSuite(model_config)
    results = full_suite.generate(
        torch.randint(0,50000,(1,512)),  # Dummy prompt tokens
        args.styles,
        num_frames=args.frames
    )
    save_video(results['styled_video'], args.output_prefix+'.mp4')
    music_to_wav(results['music'], args.output_prefix+'.wav')
    print(f"Full creative video and adaptive music saved to prefix {args.output_prefix}")
