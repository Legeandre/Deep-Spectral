# Physics-Informed Neural Networks (PINN) para Equação de Schrödinger 1D

## 1. Problema
- **Tarefa:** Resolver o problema de autovalor da Equação de Schrödinger Independente do Tempo e simular a dinâmica temporal de pacotes de onda.
- **Contexto:** Mecânica Quântica Computacional. O sistema estudado é o **Meio Oscilador Harmônico** (Half Harmonic Oscillator).
- **Métrica(s):** - Precisão dos Autovalores de Energia ($|E_{pred} - E_{teórico}|$).
  - Consistência dos observáveis dinâmicos ($\langle x \rangle, \langle p \rangle$) comparados a métodos espectrais matriciais.

## 2. Dados
- **Fonte:** Physics-Informed (Dados gerados sinteticamente durante o treino).
- **Domínio:** $x \in [0, 6]$ com Condições de Contorno de Dirichlet $\psi(0)=\psi(6)=0$.
- **Pré-processamento:** Input Scaling (Mapeamento do domínio físico para $[-1, 1]$) e Normalização da função de onda via regra de Simpson.

## 3. Modelo e Treinamento
- **Arquitetura:** MLP (Multi-Layer Perceptron) com 3 camadas ocultas de 64 neurônios.
- **Ativação:** Tanh (Tangente Hiperbólica).
- **Otimização Híbrida:**
  1. **Adam:** Exploração global do espaço de parâmetros.
  2. **L-BFGS:** Otimizador de segunda ordem para refinamento de alta precisão (até $10^{-13}$ na Loss).
- **Loss Function:** Combina resíduo da EDP, condições de contorno, normalização integral e **ortogonalidade** com estados anteriores (Gram-Schmidt via Loss).

### Como rodar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar treinamento e análise
python src/train.py --config config.yaml