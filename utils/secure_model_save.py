import torch
import safetensors.torch
import json
from pathlib import Path

LEGAL_NOTICE = """\
Copyright (C) 2025 Rudra Pandey. All rights reserved.
Contact: rudra160113.work@gmail.com

// WARNING: Use of this model is strictly limited to non-criminal, non-illegal, non-deepfake, and non-explicit purposes.\n\
// AuraGen and its weights must never be used for deepfakes, harassment, pornographic or criminal activities, or for generating content with explicit/naked images or likenesses of real people without consent. All such use is forbidden and triggers violation reporting.\
"""

def save_model_safetensors_with_notice(model, path, extra_meta=None):
    state_dict = model.state_dict()
    meta = {
        'copyright': 'Copyright (C) 2025 Rudra Pandey',
        'contact': 'rudra160113.work@gmail.com',
        'usage_restriction': 'No criminal, deepfake, or explicit use. See NOTICES.md.'
    }
    if extra_meta:
        meta.update(extra_meta)
    safetensors.torch.save_file(state_dict, path, metadata=meta)
    # Also store meta/notice as a sidecar file
    notice_path = Path(path).with_suffix('.notice.txt')
    notice_path.write_text(LEGAL_NOTICE)

def save_model_binary_with_notice(model, path):
    torch.save({
        'state_dict': model.state_dict(),
        'legal_notice': LEGAL_NOTICE
    }, path)
    notice_path = Path(path).with_suffix('.notice.txt')
    notice_path.write_text(LEGAL_NOTICE)
