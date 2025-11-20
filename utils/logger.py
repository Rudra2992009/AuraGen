import logging
from pathlib import Path
from datetime import datetime

class AuraLogger:
    """Centralized logging for AuraGen operations."""
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('AuraGen')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f'auragen_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def log_generation(self, prompt: str, duration: int, status: str):
        """Log video generation attempt."""
        self.logger.info(f"Generation | Prompt: {prompt[:50]}... | Duration: {duration}s | Status: {status}")
    
    def log_safety_block(self, prompt: str, reason: str):
        """Log safety filter blocks."""
        self.logger.warning(f"SAFETY BLOCK | Prompt: {prompt[:50]}... | Reason: {reason}")
