# Arquivo: src/utils/__init__.py

# Utilitários de Matemática
from .math_utils import torch_simpson

# Utilitários de Reprodutibilidade e Plotagem
from .reproducibility import set_seeds, setup_plots

# Gerenciadores de Lógica (Managers)
from .superposition import SuperpositionManager
from .trainer_wrapper import QuantumSystemTrainer

__all__ = [
    "torch_simpson",
    "set_seeds",
    "setup_plots",
    "SuperpositionManager",
    "QuantumSystemTrainer"
]