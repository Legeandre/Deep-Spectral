import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class SuperpositionManager:
    """
    Gerenciador de superposição de autoestados (PINN) com:
    - Separação correta de pastas (Data vs Figures)
    - Correção de fase (gauge fixing)
    - Cálculo preciso de elementos de matriz via Simpson
    """

    def __init__(self, solvers_list, coeffs_list):
        self.version = "PINN_Matrix_Solver_v2.2_DataFixed"
        # Definimos apenas o NOME do arquivo base, sem a pasta.
        # As pastas serão definidas dinamicamente nos métodos.
        self.base_name = "PINN_Dynamics" 
        self.num_levels = len(solvers_list)

        # 1) Normalização dos coeficientes
        c = np.array(coeffs_list, dtype=np.complex128)
        norm_c = np.sqrt(np.sum(np.abs(c)**2))
        self.coeffs = c / (norm_c if norm_c > 0 else 1.0)

        # Energia
        self.energies = np.array([s.get_energy() for s in solvers_list], dtype=np.float64)

        print(">>> [Init] Extraindo dados e corrigindo FASE (Gauge Fixing)...")

        # Dados espaciais
        self.x_arr, _, _ = solvers_list[0].get_state_data()
        self.x_arr = np.asarray(self.x_arr, dtype=np.float64)
        
        psis, psi_primes = [], []

        for s in solvers_list:
            x_np, psi_np, psi_x_np = s.get_state_data()

            # --- GAUGE FIXING ROBUSTO (Correção do Sinal) ---
            # Em vez de olhar o máximo global (que pode ser o segundo lóbulo negativo),
            # olhamos para o primeiro ponto onde a função se torna significativa.
            # Para o meio oscilador, a função sempre deve "nascer" positiva a partir de x=0.
            
            # 1. Acha o valor máximo absoluto para referência
            max_abs = np.max(np.abs(psi_np))
            
            # 2. Acha o índice do primeiro ponto que excede 10% do máximo
            # Isso ignora o zero exato na borda e pequenos ruídos iniciais
            significant_indices = np.where(np.abs(psi_np) > 0.1 * max_abs)[0]
            
            if len(significant_indices) > 0:
                first_idx = significant_indices[0]
                val_at_start = psi_np[first_idx]
                
                # Se o primeiro lóbulo for negativo, inverte TUDO
                if val_at_start < 0:
                    # print(f"    -> Invertendo sinal do estado (E={s.get_energy():.2f}) para alinhar fase.")
                    psi_np = -psi_np
                    psi_x_np = -psi_x_np
            
            psis.append(psi_np)
            psi_primes.append(psi_x_np)

        self.psi_matrix = np.column_stack(psis)

        # 2) Matrizes
        self.X_matrix  = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.X2_matrix = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.P_matrix  = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.P2_matrix = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)

        print(">>> [Init] Calculando Elementos de Matriz (Simpson)...")

        def integrate_complex(y, x):
            y = np.asarray(y, dtype=np.complex128)
            x = np.asarray(x, dtype=np.float64)
            real = simpson(np.real(y), x=x)
            imag = simpson(np.imag(y), x=x)
            return real + 1j * imag

        for n in range(self.num_levels):
            psi_n, dpsi_n = psis[n], psi_primes[n]
            for m in range(self.num_levels):
                psi_m, dpsi_m = psis[m], psi_primes[m]

                # Posição
                self.X_matrix[n, m] = integrate_complex(np.conj(psi_n) * self.x_arr * psi_m, self.x_arr)
                self.X2_matrix[n, m] = integrate_complex(np.conj(psi_n) * (self.x_arr**2) * psi_m, self.x_arr)
                
                # Momento
                self.P_matrix[n, m] = -1j * integrate_complex(np.conj(psi_n) * dpsi_m, self.x_arr)
                self.P2_matrix[n, m] = integrate_complex(np.conj(dpsi_n) * dpsi_m, self.x_arr)

        # Hermitização
        def hermitize(M): return 0.5 * (M + M.conjugate().T)
        self.X_matrix = hermitize(self.X_matrix)
        self.X2_matrix = hermitize(self.X2_matrix)
        self.P_matrix = hermitize(self.P_matrix)
        self.P2_matrix = hermitize(self.P2_matrix)
        print(">>> [Init] Pronto.")

    def _get_expectation_at_t(self, t, OperatorMatrix):
        coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
        val = np.vdot(coeffs_t, OperatorMatrix @ coeffs_t)
        return float(np.real(val))

    # --- UTILITÁRIOS DE PATH ---
    def _get_data_path(self, suffix):
        """Gera caminho para salvar .txt em results/simulations/"""
        folder = os.path.join("results", "simulations") 
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.base_name}{suffix}")

    def _get_fig_path(self, suffix):
        """Gera caminho para salvar .png em results/figures/"""
        folder = os.path.join("results", "figures")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.base_name}{suffix}")

    # =========================================================================
    # POSIÇÃO (Calcula em 'data', Plota em 'figures')
    # =========================================================================
    def expected_position_and_uncertainty_are_calculated(self, t_range, coefficients=None, num_points=300):
        if coefficients is not None:
            c = np.array(coefficients, dtype=np.complex128)
            self.coeffs = c / np.sqrt(np.sum(np.abs(c)**2))

        t0, t1 = t_range
        times = np.linspace(t0, t1, num_points, dtype=np.float64)
        
        # SALVA EM DATA
        filename = self._get_data_path("_PositionUncertainty.txt")

        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Filename : {filename}\n")
            f.write(f"# Version  : {self.version}\n")
            f.write(f"# Start    : {datetime.datetime.now().strftime('%c')}\n")
            f.write("# (1) time\t(2) <x>\t(3) <x>-dx\t(4) <x>+dx\n")
            f.write("# " + "="*80 + "\n")

        with open(filename, "a", encoding='utf-8') as f:
            for t in times:
                val_x  = self._get_expectation_at_t(t, self.X_matrix)
                val_x2 = self._get_expectation_at_t(t, self.X2_matrix)
                uncert = np.sqrt(max(0.0, val_x2 - val_x**2))
                f.write(f"{t:.10f}\t{val_x:.10f}\t{(val_x - uncert):.10f}\t{(val_x + uncert):.10f}\n")

        print(f"Dados salvos em: {filename}")

    def expected_position_and_uncertainty_are_plotted(self):
        # LÊ DE DATA
        filename = self._get_data_path("_PositionUncertainty.txt")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {filename}")
        
        data = np.loadtxt(filename)
        if data.ndim == 1: data = data.reshape(1, -1)

        time_vals, expected_vals = data[:, 0], data[:, 1]
        minus_vals, plus_vals = data[:, 2], data[:, 3]

        plt.figure(constrained_layout=True)
        #plt.fill_between(time_vals, minus_vals, plus_vals, color='gray', alpha=0.2)
        plt.plot(time_vals, minus_vals, color='gray', label=r"$\langle x \rangle - \sigma_x$")
        plt.plot(time_vals, expected_vals, color='black', label=r"$\langle x \rangle$")
        plt.plot(time_vals, plus_vals, color='gray', label=r"$\langle x \rangle + \sigma_x$")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Expected Position and Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
        
        # SALVA EM FIGURES
        plotname = self._get_fig_path("_ExpectedPositionAndUncertainty.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()
        print(f"Gráfico salvo em: {plotname}")

    # =========================================================================
    # MOMENTO (Calcula em 'data', Plota em 'figures')
    # =========================================================================
    def expected_momentum_and_uncertainty_are_calculated(self, t_range, coefficients=None, num_points=300):
        if coefficients is not None:
            c = np.array(coefficients, dtype=np.complex128)
            self.coeffs = c / np.sqrt(np.sum(np.abs(c)**2))

        t0, t1 = t_range
        times = np.linspace(t0, t1, num_points, dtype=np.float64)
        
        # SALVA EM DATA
        filename = self._get_data_path("_MomentumUncertainty.txt")

        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Filename : {filename}\n")
            f.write(f"# Version  : {self.version}\n")
            f.write(f"# Start    : {datetime.datetime.now().strftime('%c')}\n")
            f.write("# (1) time\t(2) <p>\t(3) <p>-dp\t(4) <p>+dp\n")
            f.write("# " + "="*80 + "\n")

        with open(filename, "a", encoding='utf-8') as f:
            for t in times:
                val_p  = self._get_expectation_at_t(t, self.P_matrix)
                val_p2 = self._get_expectation_at_t(t, self.P2_matrix)
                uncert = np.sqrt(max(0.0, val_p2 - val_p**2))
                f.write(f"{t:.10f}\t{val_p:.10f}\t{(val_p - uncert):.10f}\t{(val_p + uncert):.10f}\n")

        print(f"Dados salvos em: {filename}")

    def expected_momentum_and_uncertainty_are_plotted(self):
        # LÊ DE DATA
        filename = self._get_data_path("_MomentumUncertainty.txt")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {filename}")
        
        data = np.loadtxt(filename)
        if data.ndim == 1: data = data.reshape(1, -1)

        time_vals, expected_vals = data[:, 0], data[:, 1]
        minus_vals, plus_vals = data[:, 2], data[:, 3]

        plt.figure(constrained_layout=True)
        #plt.fill_between(time_vals, minus_vals, plus_vals, color='gray', alpha=0.2)
        plt.plot(time_vals, minus_vals, color='gray', label=r"$\langle p \rangle - \sigma_p$")
        plt.plot(time_vals, expected_vals, color='black', label=r"$\langle p \rangle$")
        plt.plot(time_vals, plus_vals, color='gray', label=r"$\langle p \rangle + \sigma_p$")
        plt.xlabel("Time")
        plt.ylabel("Momentum")
        plt.title("Expected Momentum and Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
        
        # SALVA EM FIGURES
        plotname = self._get_fig_path("_ExpectedMomentumAndUncertainty.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()
        print(f"Gráfico salvo em: {plotname}")

    # =========================================================================
    # Visualizações Extras (Salva em Figures)
    # =========================================================================
    def plot_spacetime_density(self, t_max=10, steps=200):
        print("\nGerando Mapa Espaço-Tempo...")
        times = np.linspace(0, t_max, steps)
        E_matrix = self.energies[:, np.newaxis]
        T_matrix = times[np.newaxis, :]
        coeffs_t = self.coeffs[:, np.newaxis] * np.exp(-1j * E_matrix * T_matrix)
        
        Prob_XT = np.abs(self.psi_matrix @ coeffs_t)**2
        
        plt.figure(figsize=(10, 6))
        plt.imshow(Prob_XT, aspect='auto', origin='lower',
                   extent=[0, t_max, self.x_min, self.x_max],
                   cmap='inferno', interpolation='bilinear')
        plt.colorbar(label=r'Densidade $|\Psi(x,t)|^2$')
        plt.title("Spacetime Probability Density")
        plt.xlabel("Time")
        plt.ylabel("Position")
        
        plotname = self._get_fig_path("_Spacetime.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()

    def plot_snapshots(self, times_to_plot):
        print(f"\nGerando Snapshots para t={times_to_plot}...")
        colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))
        plt.figure(figsize=(10, 6))

        for i, t in enumerate(times_to_plot):
            coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
            prob_t = np.abs(self.psi_matrix @ coeffs_t)**2
            # Normalização simples para garantir integral visual = 1
            norm = simpson(prob_t, x=self.x_arr)
            prob_t /= norm if norm > 0 else 1.0
            
            plt.plot(self.x_arr, prob_t, lw=2, color=colors[i], label=f't={t:.1f}')

        plt.title("Probability Density Snapshots - $|\Psi(x,t)|^2$")
        plt.xlabel("Position")
        plt.ylabel(r"$|\Psi(x,t)|^2$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plotname = self._get_fig_path("_Snapshots.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()


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