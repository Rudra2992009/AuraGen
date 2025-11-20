## Usage Safety
AuraGen now actively blocks prompts and references attempting to generate deepfakes or explicit/sexual content.

### Blocked Terms
Model will reject generations for prompts containing ANY of:
- deepfake
- sex video
- blowjob
- full cloth opening
- nude
- porn
- any further explicit, NSFW, or sexual phrases (see utils/safety_filter.py)

Any request containing these terms will be **blocked** and logged for security.
