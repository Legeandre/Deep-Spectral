import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class SuperpositionManager:
    """
    Gerenciador de superposição de autoestados (PINN) com:
    - Correção de fase (gauge fixing) consistente
    - Cálculo preciso de elementos de matriz via Simpson
    - Garantia de hermiticidade dos operadores (X, X^2, P, P^2)
    - Evolução temporal de valores esperados
    """

    def __init__(self, solvers_list, coeffs_list):
        self.version = "PINN_Matrix_Solver_v2.1_FixedPhase_Hermitian"
        # AJUSTE: Salva na pasta correta dentro da estrutura do projeto
        self.root_filename = "results/figures/PINN_Dynamics"
        self.num_levels = len(solvers_list)

        # 1) Normalização dos coeficientes da superposição
        c = np.array(coeffs_list, dtype=np.complex128)
        norm_c = np.sqrt(np.sum(np.abs(c)**2))
        self.coeffs = c / (norm_c if norm_c > 0 else 1.0)

        # Energia de cada nível (float64)
        self.energies = np.array([s.get_energy() for s in solvers_list], dtype=np.float64)

        print(">>> Extraindo dados e corrigindo FASE (Gauge Fixing)...")

        # Malha espacial e dados normalizados de cada solver
        self.x_arr, _, _ = solvers_list[0].get_state_data()
        self.x_arr = np.asarray(self.x_arr, dtype=np.float64)
        self.x_min, self.x_max = self.x_arr[0], self.x_arr[-1]

        psis = []
        psi_primes = []

        for s in solvers_list:
            x_np, psi_np, psi_x_np = s.get_state_data()

            # --- Correção de fase (Gauge Fixing) ---
            idx_max = np.argmax(np.abs(psi_np))
            if psi_np[idx_max] < 0:
                psi_np = -psi_np
                psi_x_np = -psi_x_np
                # print("    -> Estado com fase invertida detectado. Corrigindo sinal...")

            psis.append(psi_np)
            psi_primes.append(psi_x_np)

        # Armazena matriz de psi (para visualização futura)
        self.psi_matrix = np.column_stack(psis)

        # 2) Matrizes dos operadores
        self.X_matrix  = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.X2_matrix = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.P_matrix  = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.P2_matrix = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)

        print(">>> Calculando elementos de matriz com Simpson...")

        # Função auxiliar: integra complexo via Simpson
        def integrate_complex(y, x):
            y = np.asarray(y, dtype=np.complex128)
            x = np.asarray(x, dtype=np.float64)
            real = simpson(np.real(y), x=x)
            imag = simpson(np.imag(y), x=x)
            return real + 1j * imag

        # Monta elementos de matriz
        for n in range(self.num_levels):
            psi_n = psis[n]
            dpsi_n = psi_primes[n]

            for m in range(self.num_levels):
                psi_m = psis[m]
                dpsi_m = psi_primes[m]

                # <n|x|m>
                integrand_x = np.conjugate(psi_n) * self.x_arr * psi_m
                self.X_matrix[n, m] = integrate_complex(integrand_x, self.x_arr)

                # <n|x^2|m>
                integrand_x2 = np.conjugate(psi_n) * (self.x_arr**2) * psi_m
                self.X2_matrix[n, m] = integrate_complex(integrand_x2, self.x_arr)

                # <n|p|m>
                integrand_p = np.conjugate(psi_n) * dpsi_m
                self.P_matrix[n, m] = -1j * integrate_complex(integrand_p, self.x_arr)

                # <n|p^2|m>
                integrand_p2 = np.conjugate(dpsi_n) * dpsi_m
                self.P2_matrix[n, m] = integrate_complex(integrand_p2, self.x_arr)

        # 3) Correção de hermiticidade
        def hermitize(M):
            return 0.5 * (M + M.conjugate().T)

        self.X_matrix  = hermitize(self.X_matrix)
        self.X2_matrix = hermitize(self.X2_matrix)
        self.P_matrix  = hermitize(self.P_matrix)
        self.P2_matrix = hermitize(self.P2_matrix)

        print(">>> Pronto.")

    # ---------------------------------------------------------------------
    # Evolução temporal do valor esperado
    # ---------------------------------------------------------------------
    def _get_expectation_at_t(self, t, OperatorMatrix):
        coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
        val = np.vdot(coeffs_t, OperatorMatrix @ coeffs_t)
        return float(np.real(val))

    # ---------------------------------------------------------------------
    # Arquivos e plots de posição
    # ---------------------------------------------------------------------
    def expected_position_and_uncertainty_are_calculated(self, t_range, coefficients=None, num_points=300):
        if coefficients is not None:
            c = np.array(coefficients, dtype=np.complex128)
            norm_c = np.sqrt(np.sum(np.abs(c)**2))
            self.coeffs = c / (norm_c if norm_c > 0 else 1.0)

        t0, t1 = t_range
        times = np.linspace(t0, t1, num_points, dtype=np.float64)
        filename = f"{self.root_filename}_PositionUncertainty.txt"

        # Garante que a pasta existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Filename :          {filename}\n")
            f.write(f"# Program version :   {self.version}\n")
            f.write(f"# Start:              {datetime.datetime.now().strftime('%c')}\n")
            f.write("# (1) time\t(2) expected position\t(3) expected position - uncertainty\t(4) expected position + uncertainty\n")
            f.write("# " + "="*80 + "\n")

        with open(filename, "a", encoding='utf-8') as f:
            for t in times:
                val_x  = self._get_expectation_at_t(t, self.X_matrix)
                val_x2 = self._get_expectation_at_t(t, self.X2_matrix)
                variance = max(0.0, val_x2 - val_x**2)
                uncert = np.sqrt(variance)
                f.write(f"{t:.10f}\t{val_x:.10f}\t{(val_x - uncert):.10f}\t{(val_x + uncert):.10f}\n")

        print(f"Data saved to {filename}")

    def expected_position_and_uncertainty_are_plotted(self):
        filename = f"{self.root_filename}_PositionUncertainty.txt"
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        time_vals, expected_vals = data[:, 0], data[:, 1]
        minus_vals, plus_vals = data[:, 2], data[:, 3]

        plt.figure(constrained_layout=True)
        plt.plot(time_vals, minus_vals, color='gray', label=r"$\langle x \rangle - \sigma_x$")
        plt.plot(time_vals, expected_vals, color='black', label=r"$\langle x \rangle$")
        plt.plot(time_vals, plus_vals, color='gray', label=r"$\langle x \rangle + \sigma_x$")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Expected Position and Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
        
        plotname = f"{self.root_filename}_ExpectedPositionAndUncertainty.png"
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()

    # ---------------------------------------------------------------------
    # Arquivos e plots de momento
    # ---------------------------------------------------------------------
    def expected_momentum_and_uncertainty_are_calculated(self, t_range, coefficients=None, num_points=300):
        if coefficients is not None:
            c = np.array(coefficients, dtype=np.complex128)
            norm_c = np.sqrt(np.sum(np.abs(c)**2))
            self.coeffs = c / (norm_c if norm_c > 0 else 1.0)

        t0, t1 = t_range
        times = np.linspace(t0, t1, num_points, dtype=np.float64)
        filename = f"{self.root_filename}_MomentumUncertainty.txt"

        # Garante que a pasta existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Filename :          {filename}\n")
            f.write(f"# Program version :   {self.version}\n")
            f.write(f"# Start:              {datetime.datetime.now().strftime('%c')}\n")
            f.write("# (1) time\t(2) expected momentum\t(3) expected momentum - uncertainty\t(4) expected momentum + uncertainty\n")
            f.write("# " + "="*80 + "\n")

        with open(filename, "a", encoding='utf-8') as f:
            for t in times:
                val_p  = self._get_expectation_at_t(t, self.P_matrix)
                val_p2 = self._get_expectation_at_t(t, self.P2_matrix)
                variance = max(0.0, val_p2 - val_p**2)
                uncert = np.sqrt(variance)
                f.write(f"{t:.10f}\t{val_p:.10f}\t{(val_p - uncert):.10f}\t{(val_p + uncert):.10f}\n")

        print(f"Data saved to {filename}")

    def expected_momentum_and_uncertainty_are_plotted(self):
        filename = f"{self.root_filename}_MomentumUncertainty.txt"
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        time_vals, expected_vals = data[:, 0], data[:, 1]
        minus_vals, plus_vals = data[:, 2], data[:, 3]

        plt.figure(constrained_layout=True)
        plt.plot(time_vals, minus_vals, color='gray', label=r"$\langle p \rangle - \sigma_p$")
        plt.plot(time_vals, expected_vals, color='black', label=r"$\langle p \rangle$")
        plt.plot(time_vals, plus_vals, color='gray', label=r"$\langle p \rangle + \sigma_p$")
        plt.xlabel("Time")
        plt.ylabel("Momentum")
        plt.title("Expected Momentum and Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
        
        plotname = f"{self.root_filename}_ExpectedMomentumAndUncertainty.png"
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()

    # ---------------------------------------------------------------------
    # Evolução no tempo  para função de onda (Real)
    # ---------------------------------------------------------------------
    def plot_wavefunction_real(self, times_to_plot):
        print("\n" + "="*77)
        print(f"Generating real part of wavefunction snapshots for t={times_to_plot}...\n")

        colors = plt.cm.plasma(np.linspace(0, 1, len(times_to_plot)))
        plt.figure(figsize=(10, 6))

        for i, t in enumerate(times_to_plot):
            coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
            psi_t = self.psi_matrix @ coeffs_t

            # Normalização (usando Simpson para consistência)
            norm = simpson(np.abs(psi_t)**2, x=self.x_arr)
            psi_t /= np.sqrt(norm)

            offset = 0
            plt.plot(self.x_arr, psi_t.real + offset, lw=2, color=colors[i],
                     label=f't = {t:.4f}')

        plt.title("Wave Function - Re[ψ(x,t)]")
        plt.xlabel("Position")
        plt.ylabel("Re[ψ(x,t)]")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.grid(True, alpha=0.3)

        plotname = f"{self.root_filename}_WaveFunction.png"
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()

    # ---------------------------------------------------------------------
    # Densidade de probabilidade
    # ---------------------------------------------------------------------
    def plot_snapshots(self, times_to_plot):
        print("\n" + "="*77)
        print(f"Generating snapshots for t={times_to_plot}...\n")

        colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))
        plt.figure(figsize=(10, 6))

        for i, t in enumerate(times_to_plot):
            coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
            psi_t = self.psi_matrix @ coeffs_t
            prob_t = np.abs(psi_t)**2

            # Normalização via Simpson
            norm = simpson(prob_t, x=self.x_arr)
            prob_t /= norm

            offset = 0
            plt.plot(self.x_arr, prob_t + offset, lw=2, color=colors[i],
                     label=f't = {t:.4f}')

        plt.title("Probability Density - |Psi(x,t)|²")
        plt.xlabel("Position")
        plt.ylabel(r"$|\Psi(x,t)|²$")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.grid(True, alpha=0.3)

        plotname = f"{self.root_filename}_ProbabilityDensity.png"
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()
    
    # ---------------------------------------------------------------------
    # Posição em função do tempo - 2D
    # ---------------------------------------------------------------------
    def plot_spacetime_density(self, t_max=10, steps=500):
        print("\nGerando Mapa Espaço-Tempo...")
        times = np.linspace(0, t_max, steps)
        E_matrix = self.energies[:, np.newaxis]
        T_matrix = times[np.newaxis, :]
        coeffs_t = self.coeffs[:, np.newaxis] * np.exp(-1j * E_matrix * T_matrix)
        
        Psi_XT = self.psi_matrix @ coeffs_t
        Prob_XT = np.abs(Psi_XT)**2
        
        plt.figure(figsize=(10, 6))
        plt.imshow(Prob_XT, aspect='auto', origin='lower',
                   extent=[0, t_max, self.x_min, self.x_max],
                   cmap='inferno', interpolation='bilinear')
        plt.colorbar(label=r'Densidade $|\Psi(x,t)|^2$')
        plt.title("Spacetime Probability Density")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.savefig(f"{self.root_filename}_Spacetime.png", bbox_inches='tight')

        plotname = f"{self.root_filename}_SpaceTimeDensity.png"
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()