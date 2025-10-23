import os
import torch
import json
import yaml
from typing import Dict, Any, Optional


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(directory, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], 
                   filepath: str,
                   is_best: bool = False) -> None:
    """Save model checkpoint"""
    
    ensure_dir(os.path.dirname(filepath))
    
    # Save main checkpoint
    torch.save(state, filepath)
    
    # Save best model separately
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), 'model_best.pth')
        torch.save(state, best_path)


def load_checkpoint(filepath: str, 
                   device: str = "cpu") -> Dict[str, Any]:
    """Load model checkpoint"""
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to file"""
    
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        if filepath.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            yaml.dump(config, f, default_flow_style=False)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from file"""
    
    with open(filepath, 'r') as f:
        if filepath.endswith('.json'):
            return json.load(f)
        else:
            return yaml.safe_load(f)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get the latest checkpoint file in directory"""
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.endswith('.pth') and f != 'model_best.pth']
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])