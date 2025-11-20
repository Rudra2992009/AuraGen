import torch
import torch.nn as nn
import re
from typing import List

class AntivirusHeuristicModule(nn.Module):
    """AI-powered antivirus model for scanning, tagging, and repelling code-based threats."""
    def __init__(self, scan_dim: int = 512):
        super().__init__()
        self.scan_dim = scan_dim
        self.scanner = nn.GRU(scan_dim, scan_dim, batch_first=True)
        self.tagger = nn.Linear(scan_dim, 2)  # 0: clean, 1: malicious
        self.repeller = nn.Sequential(
            nn.Linear(scan_dim, scan_dim),
            nn.ReLU(),
            nn.Linear(scan_dim, scan_dim)
        )
    def scan_code(self, code_tokens: torch.Tensor) -> torch.Tensor:
        # Heuristic scan for suspicious patterns
        # (in deployment, feed code embeddings)
        out, _ = self.scanner(code_tokens)
        tags = self.tagger(out)
        tagged = (tags[:,:,1] > 0.5).any().item()
        return tagged
    def repel_malware(self, code_tokens: torch.Tensor) -> torch.Tensor:
        # Applies repelling transformations to neutralize malicious payload
        repelled = self.repeller(code_tokens)
        return repelled

# Example utility for keyword heuristic scan (non-deep)
MALWARE_KEYWORDS = [ 'trojan', 'virus', 'payload', 'exploit', 'ransom', 'keylogger', 'rootkit',
    'backdoor', 'worm', 'spyware', 'adware', 'botnet', 'malicious', 'obfuscate', 'inject', 'phishing']

def is_safe_code(code: str) -> bool:
    pat = re.compile('|'.join([re.escape(kw) for kw in MALWARE_KEYWORDS]), re.I)
    if pat.search(code):
        return False
    return True
