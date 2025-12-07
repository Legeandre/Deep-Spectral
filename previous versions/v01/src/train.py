import argparse
import yaml
import os
import torch
import sys 

# Adiciona o diretório "pai" (raiz do projeto) ao caminho do Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --------------------------

# Imports relativos (supondo execução da raiz com `python src/train.py`)
from src.utils.reproducibility import set_seeds, setup_plots
from src.data.physics import V_half
from src.utils.trainer_wrapper import QuantumSystemTrainer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seeds(config['experiment']['seed'])
    setup_plots()
    
    # Cria pastas de resultado
    os.makedirs("results/figures", exist_ok=True)
    
    # Configurações
    domain = tuple(config['physics']['domain'])
    layers = config['model']['layers']
    n_points = config['model']['n_points']
    
    # Trainer
    trainer = QuantumSystemTrainer(V_half, domain, layers, n_points)
    train_cfg = config['training']
    
    # Treino Sequencial
    trainer.train_states(
        n_states=train_cfg['n_states'],
        initial_E0_guess=train_cfg['initial_E0_guess'],
        delta_E_guess=train_cfg['delta_E_guess'],
        epochs_adam=train_cfg['epochs_adam'],
        epochs_lbfgs=train_cfg['epochs_lbfgs']
    )
    
    # Análise
    print("\n=== Gerando Análises ===")
    # Superposição igual (1, 1, 1) - estados equiprováveis
    coeffs = [1.0] * train_cfg['n_states']
    manager = trainer.create_superposition(coeffs)
    
    # Define local de salvamento
    manager.root_filename = "results/figures/PINN_Dynamics"
    
    # Cálculos
    manager.expected_position_and_uncertainty_are_calculated((0, 10)) # posição esperada e incerteza associada
    manager.expected_momentum_and_uncertainty_are_calculated((0, 10)) # momento esperado e incerteza associada
    
    # Plots
    manager.expected_position_and_uncertainty_are_plotted() # plota o gráfico da posição e sua incerteza
    manager.expected_momentum_and_uncertainty_are_plotted() # plota o gráfico do momento e sua incerteza
    
    manager.plot_snapshots([0.0, 1.5, 3.0]) # instantes de tempo para a densidade de probabilidade
    manager.plot_wavefunction_real(times_to_plot=[0.0, 1.0, 2.0]) # funções de onda - Re(psi(x,t))
    
    manager.plot_spacetime_density(t_max=10)


if __name__ == "__main__":
    main()