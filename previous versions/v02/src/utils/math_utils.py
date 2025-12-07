import torch

# -------------------------------------------------------------------------
# Utilidades numéricas: Simpson em PyTorch (mantém gradiente)
# -------------------------------------------------------------------------
def torch_simpson(y, x):
    """
    Integração numérica Simpson (O(h^4)) compatível com Autograd.
    """
    y = y.squeeze()
    x = x.squeeze()
    N = y.shape[0]
    
    if N < 3:
        return torch.trapz(y, x)
    
    h = x[1] - x[0]
    # Verificação de segurança para grid uniforme
    if not torch.allclose(x[2] - x[1], h, rtol=1e-5, atol=1e-8):
        return torch.trapz(y, x)
        
    if N % 2 == 0:
        # Se par, integra os N-1 primeiros com Simpson e o último intervalo com trapézio
        return torch_simpson(y[:-1], x[:-1]) + torch.trapz(y[-2:], x[-2:])
        
    odd_idx = torch.arange(1, N-1, 2, dtype=torch.int64)
    even_idx = torch.arange(2, N-2, 2, dtype=torch.int64)
    return (h / 3.0) * (y[0] + y[-1] + 4.0 * torch.sum(y[odd_idx]) + 2.0 * torch.sum(y[even_idx]))