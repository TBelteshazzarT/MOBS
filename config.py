"""
 MOBS v1.0
 Minecraft Object Boundary System
 Image Intelligence

 This script serves as a location to store project wide objects in the form of a dataclass.
 At the beginning of a script, be sure to use:

 .. code-block:: python

from config import config

"""

import os
from dataclasses import dataclass

@dataclass
class Config:
    # Dataset paths
    DATA_DIR: str = "data"
    IMAGES_DIR: str = os.path.join(DATA_DIR, "images")
    ANNOTATIONS_DIR: str = os.path.join(DATA_DIR, "annotations")

    # Model parameters
    BATCH_SIZE: int = 4
    IMAGE_SIZE: tuple = (320, 320)  # Width, Height
    NUM_CLASSES: int = 5  # diamond, gold, iron, redstone, coal

    # Training parameters
    LEARNING_RATE: float = 0.004
    MAX_EPOCHS: int = 50
    PATIENCE: int = 10  # Early stopping

    # Detection parameters
    CONFIDENCE_THRESHOLD: float = 0.5

    # Label mapping
    LABEL_MAP: dict = None

    def __post_init__(self):
        self.LABEL_MAP = {
            1: 'diamond',
            2: 'gold',
            3: 'iron',
            4: 'redstone',
            5: 'coal'
        }


# Singleton instance
config = Config()