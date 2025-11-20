# AuraGen: Long-Form AI Video Generator

**Creator & Copyright**: Rudra Pandey ([rudra160113.work@gmail.com](mailto:rudra160113.work@gmail.com))
**License**: Custom, strictly non-criminal/anti-deepfake/anti-explicit. See NOTICES.md & USAGE_SAFETY.md.

---

## Overview
AuraGen is a next-gen local video generation AI, capable of producing 1-2 hour hyper-realistic films, documentaries, creative sequences, music videos, and more—with pioneering neural chains, imagination, meta-learning blocks, antivirus safety, and full project management. Handles text prompts, reference images, and complex scene planning.

---

## Key Features
- Generate long-form (up to 2 hours) hyper-realistic videos
- Advanced realism (human-like behavior, physics, meta-learning)
- Style transfer, adaptive soundtracks, emotional narratives
- Multimodal: combines text, images, audio, and story structure
- Safety: blocks deepfakes, explicit/criminal prompts/images, auto-antivirus
- Modular Python, C++, and CLI interfaces
- Script and song planning tools, voice sync, logging, workflow management

---

## Integration Instructions

### 1. Clone & Setup
```bash
git clone https://github.com/Rudra2992009/AuraGen.git
cd AuraGen
pip install -r requirements.txt
mkdir -p weights data output projects logs
```

### 2. Project Initialization
```bash
python auragen_cli.py --prompt "A student creates an AI model" --duration 30 --project_name "first_project"
```

### 3. Secure Model Usage
All API and local integrations must use one of these access tokens:
- `rudra_qazwsxedcrfvtgbyhnujmikolp`
- `rudra_plokmijnuhbygvtfcrdxeszwaq`

Example JSON request:
```json
{
  "prompt": "A journey for justice",
  "reference_images": ["girl.jpg", "symbol.png"],
  "access_token": "rudra_qazwsxedcrfvtgbyhnujmikolp"
}
```
Can be processed by utilities/json_integration.py or via CLI.

---

## Advanced Model Selection
- AuraGen-Base (18B): Standard for HD generation, fits 6-7GB
- AuraGenAdvance18B: Extra features, realism, meta-learning
- AuraGenAdvance70B: Largest scale, for research (requires high memory)

Switch models by editing config or CLI; see scripts/estimate_parameters.py & MODEL_PARAMS.md for more.

---

## Safety & Legal Compliance
- Deepfake, nudity, explicit, criminal prompts/images blocked automatically
- All attempts logged, antivirus scan on all data/weights
- See [NOTICES.md](./NOTICES.md), [USAGE_SAFETY.md](./USAGE_SAFETY.md), and [AI_ANTIVIRUS.md](./AI_ANTIVIRUS.md)

---

## Codespaces & Cloud Usage
- Works in GitHub Codespaces with 45GB storage
- Ready for local, cloud, or clustered multi-GPU deployment
- All security features enforced

---

## Documentation & Support
- Setup: INSTALLATION.md
- Model Details: ADVANCED_18B_FEATURES.md, ADVANCED_FEATURES.md, MODEL_PARAMS.md
- API/code examples: scripts/, utils/
- Contact: rudra160113.work@gmail.com
- Attribution required on all deployments

---

## Final Notes
AuraGen sets a new standard for ethical, high-quality, long-form video generation. For creative, research, and safe enterprise use. Direct all inquiries, collaborations, or compliance reports to Rudra Pandey.
