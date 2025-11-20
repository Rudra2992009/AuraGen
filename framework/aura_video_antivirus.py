import torch
from utils.antivirus_module import AntivirusHeuristicModule, is_safe_code

class AuraVideoAntivirus:
    """Antivirus defender for AuraGen video model: scans, tags, and repels malware threats."""
    def __init__(self):
        self.antivirus = AntivirusHeuristicModule()
    def check_and_repel(self, code_str: str, code_tokens: torch.Tensor) -> str:
        # Heuristic string scan first
        if not is_safe_code(code_str):
            return "Potential malware/virus/trojan detected: scan failed. Generation blocked and code is repelled."
        # AI scan next
        tagged = self.antivirus.scan_code(code_tokens)
        if tagged:
            repelled = self.antivirus.repel_malware(code_tokens)
            return "AI identified and repelled malicious code payload."
        return "Code is clean and safe."
