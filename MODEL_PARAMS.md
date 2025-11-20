# Model Parameter Sizes

| Model             | Parameters (Approx.) |
|-------------------|---------------------|
| AuraGen-Small     | ~3B                 |
| AuraGen-Base      | ~18B                |
| AuraGen-Large     | ~120B+              |

**Note**: Parameter counts are estimates; exact value depends on final architecture (see scripts/estimate_parameters.py). Choose configuration as needed:
- `AuraGen-Small`: Lower memory, good for rapid prototyping.
- `AuraGen-Base`: Strong performance, full HD video.
- `AuraGen-Large`: Highest realism at 4K, requires high VRAM.
