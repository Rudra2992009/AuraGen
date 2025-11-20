# AuraGen - Installation & Setup Guide

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Rudra2992009/AuraGen.git
cd AuraGen
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Directories
```bash
mkdir -p weights data output projects logs
```

## Quick Start

### Basic Video Generation
```bash
python auragen_cli.py --prompt "A student building an AI model" --duration 30 --project_name "my_first_video"
```

### With Reference Images
```bash
python auragen_cli.py --prompt "A journey through space" --reference_images image1.jpg image2.jpg --duration 60
```

### Script Preparation
```bash
python scripts/prepare_script_and_songs.py
```

### Generate Secure Weights
```bash
python scripts/gen_secure_safetensor.py
```

## GitHub Codespaces Setup

1. Open repository in GitHub Codespaces
2. Storage available: 45GB
3. Install dependencies automatically
4. Run training or generation

## Safety Features

- Automatic prompt filtering (deepfake/explicit content blocked)
- Reference image validation (nudity detection)
- AI antivirus scanning for malware
- All attempts logged for security

## Project Structure

```
AuraGen/
├── model/              # Neural network architectures
├── framework/          # Complete pipelines
├── utils/              # Utilities and helpers
├── scripts/            # Helper scripts
├── weights/            # Model weights (safetensors)
├── output/             # Generated videos
├── projects/           # Project management
└── logs/               # Operation logs
```

## Contact

Copyright © 2025 Rudra Pandey
Email: rudra160113.work@gmail.com

⚠️ See NOTICES.md and USAGE_SAFETY.md for legal restrictions.
