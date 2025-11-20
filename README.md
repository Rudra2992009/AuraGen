# AuraGen

AuraGen is an advanced long-form video generation model that creates up to 2-hour realistic videos with neural chain architecture, dialogue and background music generation, all designed for fully local operation—no API keys or external services required.

## Features
- Local generation of full videos (up to 2 hours) with story planning, music, dialogues, and scene transitions.
- Highly scalable neural chain and temporal attention system supporting realistic, coherent films.
- Python module for model building/training, with C++ (`aura.cpp`) integration for local deployment.
- Data privacy: All processes run on your system. No calls to cloud APIs or third parties.

## Quick Start
Clone the repo, then build or run either the Python or C++ interface.

```sh
git clone https://github.com/Rudra2992009/AuraGen.git
cd AuraGen
# Python: Training/inference
python -m model.training
# C++ usage
# g++ aura.cpp -ltorch -o auragen
# ./auragen "<prompt>" <duration_seconds> <output_path> <path_to_model.pt>
```

## Integration via JSON
AuraGen supports simple JSON integration for trusted local automation, using predefined access tokens below. _These are useful for local trusted scripts or systems but are never sent to an external server._

**Predefined Access Tokens:**
- `rudra_qazwsxedcrfvtgbyhnujmikolp`
- `rudra_plokmijnuhbygvtfcrdxeszwaq`

Example JSON usage:
```json
{
    "prompt": "A 10th class student building world-changing AI",
    "duration": 5400,
    "access_token": "rudra_qazwsxedcrfvtgbyhnujmikolp"
}
```

These tokens allow the local integration layer in `aura.cpp` or Python scripts to recognize authorized requests. Users can use **either** token with their trusted automation workflows.

**Note:** AuraGen is designed to be run and managed locally. It is not deployed or exposed on any public website or external cloud. All model weights or data remain on the local or codespaces filesystem only.
