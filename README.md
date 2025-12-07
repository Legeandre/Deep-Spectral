# Projeto Final — Deep Learning: Dinâmica Quântica com PINNs

## 1\. Problema

  - **Tarefa:** Resolução de Equação Diferencial Parcial (EDP) via Redes Neurais Informadas pela Física (PINNs) — Problema de Autovalor.
  - **Contexto:** Mecânica Quântica Não-Relativística. O objetivo é resolver a Equação de Schrödinger para o **Oscilador Meio-Harmônico** (*Half-Harmonic Oscillator*) e simular sua dinâmica temporal.
  - **Métrica(s):**
      - **Função de Perda Física ($\mathcal{L}$):** Resíduo da Equação de Schrödinger + Condições de Contorno + Normalização + Ortogonalidade.
      - **Erro Absoluto ($\Delta$):** Comparação quantitativa dos observáveis (Posição $\langle x \rangle$ e Momento $\langle p \rangle$) contra um *baseline* numérico (Método Espectral).
      - **Fidelidade Visual:** Coerência dos mapas de densidade de probabilidade espaço-temporal $|\Psi(x,t)|^2$.

## 2\. Dados

Como se trata de um método *mesh-free* (livre de malha), não há um dataset externo estático. Os dados são gerados dinamicamente durante o treinamento.

  - **Fonte:** Amostragem estocástica do domínio espacial.
  - **Domínio:** $x \in [0, 6]$ (Unidades Atômicas).
  - **Tamanho:** $N_{col} = 10.000$ pontos de colação gerados a cada época para cálculo do resíduo da EDP.
  - **Pré-processamento:** Não aplicável (geração *on-the-fly*).
  - **Geração:** Controlada pelos parâmetros em `config.yaml`.

## 3\. Modelo e Treinamento

O projeto utiliza uma abordagem híbrida: **PINN** para encontrar os estados estacionários e **Mecânica Matricial** para a evolução temporal.

  - **Arquitetura:** Perceptron Multicamadas (MLP) totalmente conectado.
      - **Topologia:** `[1, 64, 64, 64, 1]` (Input $\to$ 3 camadas ocultas $\to$ Output).
      - **Ativação:** `Tanh` (Tangente Hiperbólica) para garantir suavidade ($C^\infty$) e diferenciabilidade necessária para o cálculo do Laplaciano.
  - **Hiperparâmetros Chave:**
      - **Otimizador 1 (Exploração):** Adam (`lr=5e-5`, 7000 épocas).
      - **Otimizador 2 (Refinamento):** L-BFGS (`lr=1.0`, 6000 iterações, tolerância `1e-9`).
      - **Pesos da Loss ($\lambda$):** PDE=1.0, BC=100.0, Norm=100.0, Ortho=500.0.
  - **Config:** Todos os parâmetros estão centralizados em `config.yaml`.
  - **Reprodutibilidade:** Semente aleatória fixada (ex: `seed: 42`) via `torch.manual_seed` e `np.random.seed`.

### Como rodar

```bash
# 1. Criar ambiente virtual (opcional)
conda create -n pinn_quantum python=3.9
conda activate pinn_quantum

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Treinar o modelo e gerar simulações
# O script treina sequencialmente os estados n=0,1,2 e gera os plots
python src/train.py --config config.yaml

# 4. Resultados
# Verifique a pasta results/figures para os gráficos gerados
```

## 4\. Resultados

A solução proposta foi capaz de reproduzir a dinâmica do sistema com alta fidelidade.

  - **Tabela de Métricas (Final da Simulação $t=10$):**

| Observável | Erro Máx. Absoluto (vs Espectral) | Erro Relativo Aprox. |
| :--- | :---: | :---: |
| Posição $\langle x \rangle$ | \~0.10 u.a. | \< 5% |
| Momento $\langle p \rangle$ | \~0.23 u.a. | \< 7% |

  - **Figuras Geradas:**

      - **Funções de Onda:** Visualização dos estados estacionários $\psi_0, \psi_1, \psi_2$.
      - **Densidade Espaço-Tempo:** Heatmap da evolução $|\Psi(x,t)|^2$ mostrando o "batimento" do pacote de onda.
      - **Observáveis:** Gráficos de $\langle x \rangle(t)$ e $\langle p \rangle(t)$ comparados com o baseline, incluindo faixas de incerteza ($\sigma_x, \sigma_p$).

  - **Comparação com Baseline:**

      - O método foi validado contra um *solver* espectral clássico (baseado em diferenças finitas e `scipy.linalg`).
      - Implementou-se um algoritmo de **Gauge Fixing** (Correção de Fase) e ajuste de sinal na matriz de momento para alinhar as convenções físicas entre a rede neural estocástica e o solver determinístico.

## 5\. Limitações e Próximos Passos

  - **Limitações:** O método atual é mais custoso computacionalmente que métodos espectrais simples para problemas 1D de baixa energia, devido ao tempo de treinamento da rede. A precisão depende do ajuste fino dos pesos $\lambda$ da função de perda.
  - **Próximos Passos:**
      - Generalizar o código para aceitar potenciais arbitrários ($V(x)$) definidos pelo usuário.
      - Estender a aplicação para problemas 2D onde métodos de malha sofrem com a maldição da dimensionalidade.

## 6\. Autores

  - **Nome:** Vagner Jandre Monteiro
  - **Matrícula:** DO2411592
  - **Curso:** PPGMC-DO / IPRJ-UERJ