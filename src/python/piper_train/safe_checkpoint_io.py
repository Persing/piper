import torch
from pathlib import PosixPath
from pytorch_lightning.plugins.io import TorchCheckpointIO

class SafeCheckpointIO(TorchCheckpointIO):
    """Custom checkpoint loader with PosixPath support"""
    def load_checkpoint(self, checkpoint_path, map_location=None):
        # Allow PosixPath in checkpoints
        with torch.serialization.safe_globals([PosixPath]):
            return torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=False  # Only use with trusted checkpoints!
            )