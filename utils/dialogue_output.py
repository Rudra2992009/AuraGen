import torch
import numpy as np
import os

def save_dialogue(dialogue_tensor, filename):
    """Save dialogue (tokens) as simple text."""
    # Placeholder: token ids to string
    dialogue_str = ' '.join([str(int(tok)) for tok in dialogue_tensor.view(-1)])
    with open(filename, 'w') as f:
        f.write(dialogue_str)
if __name__ == "__main__":
    dialogue_tensor = torch.randint(0, 50000, (1,32))
    save_dialogue(dialogue_tensor, 'output/dummy_dialogue.txt')
    print("Dialogue saved: output/dummy_dialogue.txt")
