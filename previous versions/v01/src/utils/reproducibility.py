import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def set_seeds(seed=42):
    """Fixa sementes e define precisão dupla (float64)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # CRÍTICO: Autovalores precisos exigem float64
    torch.set_default_dtype(torch.float64)
    print(f"Seeds fixadas em: {seed} | Dtype: float64")

def setup_plots():
    plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})