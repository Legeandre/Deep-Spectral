import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from scipy.integrate import simpson
from src.utils.math_utils import torch_simpson

class _PINNModel(nn.Module):
    def __init__(self, layers):
        super(_PINNModel, self).__init__()
        self.depth = len(layers) - 1
        layer_list = []
        for i in range(self.depth):
            layer_list.append((f'layer_{i}', nn.Linear(layers[i], layers[i+1])))
            if i < self.depth - 1:
                layer_list.append((f'activation_{i}', nn.Tanh()))
        self.net = nn.Sequential(OrderedDict(layer_list))
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class PINNSolver1D:
    def __init__(self, V_func, x_domain, layers, n_points, initial_E_guess=0.0, previous_solvers=None):
        self.V_func = V_func
        self.x_min, self.x_max = x_domain
        self.n_points = n_points
        self.previous_solvers = previous_solvers if previous_solvers else []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x = torch.linspace(self.x_min, self.x_max, n_points, dtype=torch.float64, device=self.device).unsqueeze(1)
        self.x.requires_grad_(True)
        self.domain_len = self.x_max - self.x_min
        self.scale_factor = 2.0 / self.domain_len

        self.model = _PINNModel(layers).double().to(self.device)
        self.E = torch.nn.Parameter(torch.tensor([initial_E_guess], dtype=torch.float64, device=self.device))

        self.lambda_bc = 100.0
        self.lambda_norm = 100.0
        self.lambda_ortho = 500.0
        self.loss_history = []
        self.energy_history = []
        self.trained = False

    def _loss_function(self):
        x_norm = 2.0 * (self.x - self.x_min) / self.domain_len - 1.0
        psi = self.model(x_norm)
        
        dpsi_dxnorm = torch.autograd.grad(psi, x_norm, torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        psi_x = dpsi_dxnorm * self.scale_factor
        d2psi_dxnorm2 = torch.autograd.grad(dpsi_dxnorm, x_norm, torch.ones_like(dpsi_dxnorm), create_graph=True, retain_graph=True)[0]
        psi_xx = d2psi_dxnorm2 * (self.scale_factor ** 2)

        V_x = self.V_func(self.x)
        residual = -psi_xx + (V_x * psi) - (self.E * psi)
        loss_physics = torch.mean(residual**2)

        x_bc_phys = torch.tensor([[self.x_min], [self.x_max]], dtype=torch.float64, device=self.device)
        x_bc_norm = 2.0 * (x_bc_phys - self.x_min) / self.domain_len - 1.0
        psi_bc = self.model(x_bc_norm)
        loss_bc = torch.mean(psi_bc**2)

        psi_sq = psi.squeeze()**2
        x_flat = self.x.squeeze()
        integral_norm = torch_simpson(psi_sq, x_flat)
        loss_norm = (integral_norm - 1.0)**2

        loss_ortho = 0.0
        if self.previous_solvers:
            with torch.no_grad():
                psi_prev_all = [prev.model(x_norm) for prev in self.previous_solvers]
            for psi_prev in psi_prev_all:
                overlap = torch_simpson((psi.squeeze() * psi_prev.squeeze()), x_flat)
                loss_ortho = loss_ortho + overlap**2

        return loss_physics + self.lambda_bc * loss_bc + self.lambda_norm * loss_norm + self.lambda_ortho * loss_ortho

    def train(self, epochs_adam=10000, epochs_lbfgs=8000):
        self.trained = False
        print("="*80)
        print(f"Iniciando Treinamento Otimizado. E_chute={self.E.item():.15f}")
        
        # --- FASE 1: Adam ---
        optimizer_adam = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': 5e-5},
            {'params': [self.E],                'lr': 5e-2}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=1000)

        print(f"\n[Fase 1] Adam (Max {epochs_adam} epochs)...")
        for i in range(epochs_adam):
            optimizer_adam.zero_grad()
            loss = self._loss_function()
            loss.backward()
            optimizer_adam.step()
            scheduler.step(loss)

            if i % 500 == 0:
                print(f"Ep {i} | Loss {loss.item():.2e} | E = {self.E.item():.6f}")
            if loss.item() < 1e-6 and i > 4000:
                print("-> Adam convergiu cedo!")
                break

        # --- FASE 2: LBFGS ---
        print(f"\n[Fase 2] LBFGS (Refinamento)...")
        optimizer_lbfgs = torch.optim.LBFGS(
            list(self.model.parameters()) + [self.E],
            lr=1.0, 
            max_iter=epochs_lbfgs,
            max_eval=int(epochs_lbfgs*1.25),
            history_size=500,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe"
        )

        pbar = 0
        best_loss = float("inf")
        no_improve_count = 0
        
        def closure():
            nonlocal pbar, best_loss, no_improve_count
            optimizer_lbfgs.zero_grad()
            loss = self._loss_function()
            loss.backward()
            self.loss_history.append(loss.item())
            self.energy_history.append(self.E.item())
            
            if abs(loss.item() - best_loss) < 1e-13:
                no_improve_count += 1
            else:
                best_loss = loss.item()
                no_improve_count = 0
            
            if pbar % 200 == 0:
                print(f"LBFGS It {pbar}: Loss {loss.item():.2e} | E = {self.E.item():.9f}")
            pbar += 1
            
            if no_improve_count >= 1000:
                raise StopIteration
            return loss

        try:
            optimizer_lbfgs.step(closure)
        except StopIteration:
            print("LBFGS convergiu (loss estável).")

        self.trained = True
        print(f"-> FINALIZADO. Energia Final: {self.E.item():.9f}")

    def get_state_data(self):
        if not self.x.requires_grad: self.x.requires_grad_(True)
        x_norm = 2.0 * (self.x - self.x_min) / self.domain_len - 1.0
        psi_raw = self.model(x_norm)
        grads = torch.autograd.grad(psi_raw, x_norm, torch.ones_like(psi_raw), create_graph=True)[0]
        psi_x_raw = grads * self.scale_factor
        
        psi_np = psi_raw.detach().cpu().numpy().flatten()
        psi_x_np = psi_x_raw.detach().cpu().numpy().flatten()
        x_np = self.x.detach().cpu().numpy().flatten()
        
        integral = simpson(psi_np**2, x_np)
        A = 1.0 / np.sqrt(integral) if integral > 0 else 1.0
        return x_np, psi_np * A, psi_x_np * A
    
    def get_energy(self):
        return self.E.item() if self.trained else None