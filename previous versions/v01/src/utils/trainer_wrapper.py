from src.models.pinn_module import PINNSolver1D
from src.utils.superposition import SuperpositionManager

class QuantumSystemTrainer:
    def __init__(self, V_func, domain, layers, n_points):
        self.V_func = V_func
        self.domain = domain
        self.layers = layers
        self.n_points = n_points
        self.solvers = []

    def train_states(self, n_states, initial_E0_guess, delta_E_guess, epochs_adam, epochs_lbfgs):
        print(f"INICIANDO SISTEMA QUÂNTICO: Buscando {n_states} estados.\n")
        current_E_guess = initial_E0_guess

        for n in range(n_states):
            print(f"\n>>> TREINANDO ESTADO N={n} (Chute E ~ {current_E_guess:.6f}) <<<")
            solver = PINNSolver1D(
                self.V_func, self.domain, self.layers, self.n_points,
                initial_E_guess=current_E_guess,
                previous_solvers=self.solvers[:]
            )
            solver.train(epochs_adam=epochs_adam, epochs_lbfgs=epochs_lbfgs)
            self.solvers.append(solver)
            
            final_E = solver.get_energy()
            if len(self.solvers) >= 2:
                delta_E_guess = self.solvers[-1].get_energy() - self.solvers[-2].get_energy()
            current_E_guess = final_E + delta_E_guess

    def create_superposition(self, coeffs):
        if len(coeffs) != len(self.solvers):
            coeffs = coeffs + [0.0]*(len(self.solvers) - len(coeffs))
        return SuperpositionManager(self.solvers, coeffs)