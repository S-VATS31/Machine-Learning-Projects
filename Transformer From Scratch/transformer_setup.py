import logging
import math
import torch
from dataclasses import dataclass
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import List, Tuple, Optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logging.basicConfig(
    level = logging.DEBUG, # Detailed info on bugs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_debug.log', mode="w") # Save to file
    ]
)

# Create logger object
logger = logging.getLogger(__name__)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
