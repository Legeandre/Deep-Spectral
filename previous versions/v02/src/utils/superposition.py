import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class SuperpositionManager:
    """
    Gerenciador de superposição de autoestados (PINN)
    
    Características:
    - Gauge Fixing: Impõe paridade (-1)^n para alinhar fase com métodos espectrais.
    - Matrix Mechanics: Pré-calcula <n|O|m> para evolução temporal exata.
    - Precisão: Usa regra de Simpson (ordem 4) para minimizar ruído numérico.
    """

    def __init__(self, solvers_list, coeffs_list):
        self.version = "PINN_Matrix_Solver_v2.3_Final"
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
        
        self.x_min = self.x_arr[0]
        self.x_max = self.x_arr[-1]

        psis, psi_primes = [], []

        for n, s in enumerate(solvers_list):
            x_np, psi_np, psi_x_np = s.get_state_data()

            # --- GAUGE FIXING (Correção de Sinal) ---
            # Para alinhar com a referência espectral do Oscilador Harmônico,
            # forçamos a paridade do sinal inicial: (+, -, +, -, ...)
            max_abs = np.max(np.abs(psi_np))
            # Pega o primeiro ponto que não é zero (acima de 5% do pico)
            sig_indices = np.where(np.abs(psi_np) > 0.05 * max_abs)[0]
            
            if len(sig_indices) > 0:
                first_idx = sig_indices[0]
                current_sign = np.sign(psi_np[first_idx])
                
                # Regra: Par (n=0,2) -> Positivo | Ímpar (n=1,3) -> Negativo
                target_sign = 1.0 if (n % 2 == 0) else -1.0
                
                if current_sign != target_sign:
                    psi_np = -psi_np
                    psi_x_np = -psi_x_np
            
            psis.append(psi_np)
            psi_primes.append(psi_x_np)

        self.psi_matrix = np.column_stack(psis)

        # 2) Matrizes (Pré-cálculo)
        self.X_matrix  = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.X2_matrix = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.P_matrix  = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)
        self.P2_matrix = np.zeros((self.num_levels, self.num_levels), dtype=np.complex128)

        print(">>> [Init] Calculando Elementos de Matriz (Simpson)...")

        # Função auxiliar de integração
        def integrate(y, x):
            return simpson(y, x=x)

        for n in range(self.num_levels):
            psi_n, dpsi_n = psis[n], psi_primes[n]
            for m in range(self.num_levels):
                psi_m, dpsi_m = psis[m], psi_primes[m]

                # Integração numérica de alta precisão
                # Posição
                self.X_matrix[n, m]  = integrate(np.conj(psi_n) * self.x_arr * psi_m, self.x_arr)
                self.X2_matrix[n, m] = integrate(np.conj(psi_n) * (self.x_arr**2) * psi_m, self.x_arr)
                
                # Momento (-1j * integral da derivada)
                self.P_matrix[n, m]  = -1j * integrate(np.conj(psi_n) * dpsi_m, self.x_arr)
                self.P2_matrix[n, m] = integrate(np.conj(dpsi_n) * dpsi_m, self.x_arr)

        # 3) Hermitização (Limpeza de ruído imaginário na diagonal)
        def hermitize(M): return 0.5 * (M + M.conjugate().T)
        self.X_matrix  = hermitize(self.X_matrix)
        self.X2_matrix = hermitize(self.X2_matrix)
        self.P_matrix  = hermitize(self.P_matrix)
        self.P2_matrix = hermitize(self.P2_matrix)
        print(">>> [Init] Pronto.")

    # --- CORE: Dinâmica Analítica ---
    def _get_expectation_at_t(self, t, OperatorMatrix):
        coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
        val = np.vdot(coeffs_t, OperatorMatrix @ coeffs_t)
        return float(np.real(val))

    # --- UTILITÁRIOS DE PATH ---
    def _get_data_path(self, suffix):
        folder = os.path.join("results", "simulations")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.base_name}{suffix}")

    def _get_fig_path(self, suffix):
        folder = os.path.join("results", "figures")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.base_name}{suffix}")

    # =========================================================================
    # POSIÇÃO
    # =========================================================================
    def expected_position_and_uncertainty_are_calculated(self, t_range, coefficients=None, num_points=300):
        if coefficients is not None:
            c = np.array(coefficients, dtype=np.complex128)
            norm_c = np.sqrt(np.sum(np.abs(c)**2))
            self.coeffs = c / (norm_c if norm_c > 0 else 1.0)

        t0, t1 = t_range
        times = np.linspace(t0, t1, num_points, dtype=np.float64)
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
        print(f"Dados Posição salvos.")

    def expected_position_and_uncertainty_are_plotted(self):
        filename = self._get_data_path("_PositionUncertainty.txt")
        if not os.path.exists(filename): raise FileNotFoundError(filename)
        data = np.loadtxt(filename)
        if data.ndim == 1: data = data.reshape(1, -1)

        plt.figure(constrained_layout=True)
        #plt.fill_between(data[:,0], data[:,2], data[:,3], color='gray', alpha=0.2)
        plt.plot(data[:,0], data[:,2], color='gray', label=r"$\langle x \rangle - \sigma_x$")
        plt.plot(data[:,0], data[:,1], color='black', label=r"$\langle x \rangle$")
        plt.plot(data[:,0], data[:,3], color='gray', label=r"$\langle x \rangle + \sigma_x$")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Expected Position and Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
        
        plotname = self._get_fig_path("_ExpectedPositionAndUncertainty.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()

    # =========================================================================
    # MOMENTO
    # =========================================================================
    def expected_momentum_and_uncertainty_are_calculated(self, t_range, coefficients=None, num_points=300):
        if coefficients is not None:
            c = np.array(coefficients, dtype=np.complex128)
            norm_c = np.sqrt(np.sum(np.abs(c)**2))
            self.coeffs = c / (norm_c if norm_c > 0 else 1.0)

        t0, t1 = t_range
        times = np.linspace(t0, t1, num_points, dtype=np.float64)
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
        print(f"Dados Momento salvos.")

    def expected_momentum_and_uncertainty_are_plotted(self):
        filename = self._get_data_path("_MomentumUncertainty.txt")
        if not os.path.exists(filename): raise FileNotFoundError(filename)
        data = np.loadtxt(filename)
        if data.ndim == 1: data = data.reshape(1, -1)

        plt.figure(constrained_layout=True)
        #plt.fill_between(data[:,0], data[:,2], data[:,3], color='gray', alpha=0.2)
        plt.plot(data[:,0], data[:,2], color='gray', label=r"$\langle p \rangle - \sigma_p$")
        plt.plot(data[:,0], data[:,1], color='black', label=r"$\langle p \rangle$")
        plt.plot(data[:,0], data[:,3], color='gray', label=r"$\langle p \rangle + \sigma_p$")
        plt.xlabel("Time")
        plt.ylabel("Momentum")
        plt.title("Expected Momentum and Uncertainty")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left")
        
        plotname = self._get_fig_path("_ExpectedMomentumAndUncertainty.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()

    # ---------------------------------------------------------------------
    # Evolução no tempo  para função de onda (Real)
    # ---------------------------------------------------------------------
    def plot_wavefunction_real(self, times_to_plot):
        """
        Plota a parte REAL da função de onda Re[Psi(x,t)] em instantes específicos.
        Útil para visualizar a propagação de fase e oscilação.
        """
        print("\n" + "="*77)
        print(f"Generating real part of wavefunction snapshots for t={times_to_plot}...\n")

        # Usa mapa de cores 'plasma' para diferenciar do 'viridis' da densidade
        colors = plt.cm.plasma(np.linspace(0, 1, len(times_to_plot)))
        plt.figure(figsize=(10, 6))

        for i, t in enumerate(times_to_plot):
            # Evolução temporal dos coeficientes
            coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
            
            # Reconstrói a função de onda total
            psi_t = self.psi_matrix @ coeffs_t

            # Normalização (usando Simpson para consistência com o resto da classe)
            norm = simpson(np.abs(psi_t)**2, x=self.x_arr)
            psi_t /= np.sqrt(norm) if norm > 0 else 1.0

            offset = 0
            # Plota apenas a parte REAL
            plt.plot(self.x_arr, psi_t.real + offset, lw=2, color=colors[i],
                     label=f't = {t:.4f}')

        plt.title("Wave Function - Re[ψ(x,t)]")
        plt.xlabel("Position")
        plt.ylabel("Re[ψ(x,t)]")
        plt.legend(loc="upper left", bbox_to_anchor=(1.00, 1))
        plt.grid(True, alpha=0.3)

        # AJUSTE IMPORTANTE: Usa o helper para salvar na pasta results/figures
        plotname = self._get_fig_path("_WaveFunction.png")
        plt.savefig(plotname, bbox_inches='tight')
        print(f"Plot salvo em: {plotname}")
        plt.show()
    
    # =========================================================================
    # VISUALIZAÇÕES EXTRAS
    # =========================================================================
    def plot_spacetime_density(self, t_max=10, steps=200):
        print("\nGerando Mapa Espaço-Tempo...")
        
        times = np.linspace(0, t_max, steps)
        E_matrix = self.energies[:, np.newaxis]
        T_matrix = times[np.newaxis, :]
        coeffs_t = self.coeffs[:, np.newaxis] * np.exp(-1j * E_matrix * T_matrix)
        
        Prob_XT = np.abs(self.psi_matrix @ coeffs_t)**2
        
        plt.figure(figsize=(10, 6))
        plt.imshow(Prob_XT, aspect='auto', origin='lower', extent=[0, t_max, self.x_min, self.x_max], cmap='inferno', interpolation='bilinear')
        plt.colorbar(label=r'Densidade $|\Psi(x,t)|^2$')
        plt.title("Spacetime Probability Density")
        plt.xlabel("Time")
        plt.ylabel("Position")
        
        plotname = self._get_fig_path("_Spacetime.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()

    def plot_snapshots(self, times_to_plot):
        print(f"\nGerando Snapshots para t={times_to_plot}...")
        
        # Gera cores diferentes para cada tempo
        colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))
        
        plt.figure(figsize=(10, 6))

        # Itera sobre a lista de tempos fornecida (ex: [0.0, 1.5, 3.0])
        for i, t in enumerate(times_to_plot):
            # Evolui os coeficientes para o tempo t
            coeffs_t = self.coeffs * np.exp(-1j * self.energies * t)
            
            # Calcula densidade
            prob_t = np.abs(self.psi_matrix @ coeffs_t)**2
            
            # Normalização via Simpson
            norm = simpson(prob_t, x=self.x_arr)
            prob_t /= norm if norm > 0 else 1.0
            
            # Plota
            plt.plot(self.x_arr, prob_t, lw=2, color=colors[i], label=f't={t:.1f}')

        plt.title("Probability Density Snapshots")
        plt.xlabel("Position")
        plt.ylabel(r"$|\Psi(x,t)|^2$")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Salva o gráfico
        plotname = self._get_fig_path("_Snapshots.png")
        plt.savefig(plotname, bbox_inches='tight')
        plt.show()
        print(f"Gráfico salvo em: {plotname}")

