import logging
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logging.basicConfig(
    level = logging.DEBUG, # Detailed info on bugs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('llm_debug.log') # Save to file
    ]
)

# Create logger object
logger = logging.getLogger(__name__)
