"""
Calcula o mapa da fidelidade média de porta F1 no plano (sigma, L) para N
sítios cross-Kerr idênticos, fótons contrapropagantes, e salva os dados em
data/heatmap_n_sitios (um .npz por valor de N) para serem plotados por
heatmap_l_vs_sigma_plotar.py.

Generaliza heatmap_l_vs_sigma_calcular.py (N=2) usando a matriz S de N sítios,
a mesma de fidelidade_sigma/calcular_sigma_n_sitios.py:

    S = <ωa-|νa+><ωb-|νb+>
        - i(χγ²/π) K(νa,νb) δ(ωa+ωb-νa-νb) / [Γ(νb)Γ(νa)Γ(ωb)Γ(ωa)]
          × Σ_{j=1}^{N} A^{N-j} B^{j-1},

    A = Γ*(ωa)Γ*(νb)/[Γ(ωa)Γ(νb)] e^{i(ωa+νb)L}
    B = Γ*(ωb)Γ*(νa)/[Γ(ωb)Γ(νa)] e^{i(ωb+νa)L}
    <ω-|ν+> = e^{i(N-1)ωL} (Γ*(ω)/Γ(ω))^N δ(ω-ν)

A fidelidade é F1(phi) = [6 + 3 Re(e^{i phi} F) + |F|^2] / 10, onde F é a
amplitude de sobreposição. São salvos F1(phi = pi), que é a porta CZ, e
F1(phi_opt); F, |F|, Re F e Im F seguem salvos para diagnóstico.
"""
import os

import numpy as np
from scipy.integrate import simpson
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------
# Parâmetros físicos fixos
# ----------------------------------------------------------------------
N_values = [4]        # números de sítios (um .npz por valor)
gamma = 1.0           # taxa de decaimento (igual em todos os sítios)
chi = 1000.0          # acoplamento não-linear (igual em todos os sítios)
Delta = 0.0           # dessintonia (igual em todos os sítios)
omega0 = 0.0          # frequência central do pacote, relativa a Delta
phi = np.pi           # fase usada no cálculo de F1(phi)

omega_c = Delta + omega0  # frequência central efetiva (derivada, não editar)

# ----------------------------------------------------------------------
# Varredura no plano (sigma, L)
# ----------------------------------------------------------------------
sigma_min = 0.01      # menor largura do pacote (eixo x, escala log)
sigma_max = 10.0      # maior largura do pacote
n_sigma = 200         # número de colunas do mapa

L_min = 0.0           # menor separação entre os sítios (eixo y, escala linear)
L_max = 1             # maior separação
n_L = 200             # número de linhas do mapa

# ----------------------------------------------------------------------
# Parâmetros numéricos da integração
# ----------------------------------------------------------------------
k_sigma = 9.0         # metade da janela de integração, em unidades de sigma
pts_per_width = 14    # pontos por menor escala relevante, min(sigma, gamma, 1/(N L))
n_min = 2001          # número mínimo de pontos na grade
n_max = 600001        # teto de pontos (proteção de tempo/memória)

# ----------------------------------------------------------------------
# Parâmetros de saída
# ----------------------------------------------------------------------
outdir = os.path.join('data', 'heatmap_n_sitios')  # pasta onde os dados serão salvos
quiet = False         # se True, não imprime o progresso da varredura


def outfile_name(N):
    """Nome do .npz de saída para um dado número de sítios N."""
    return f'heatmap_N{N}_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}.npz'


def Gamma(w):
    return gamma / 2 + 1j * (Delta - w)


def t_site(w):
    """Fase de espalhamento de um único sítio (|t|=1)."""
    return np.conj(Gamma(w)) / Gamma(w)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def correction(w, N, L):
    """Remove a amplitude linear de fóton único <ω-|ω+> = e^{i(N-1)ωL} t(ω)^N."""
    return np.conj(t_site(w)) ** N * np.exp(-1j * (N - 1) * w * L)


def grade(sigma, N, L):
    """Grade de frequências e seu espaçamento.

    A janela é fixada só por sigma (as quatro gaussianas confinam o
    integrando). O espaçamento resolve a menor escala presente: sigma, gamma
    e, para L != 0, o período das fases e^{iωL} acumuladas ao longo dos N sítios.
    """
    half = k_sigma * sigma
    scales = [sigma, gamma]
    if L != 0:
        scales.append(1.0 / (N * abs(L)))
    n = int(2 * half / (min(scales) / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1   # ímpar: Simpson com nº par de intervalos
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


def compute_F(sigma, N, L):
    """
    F = 1 + G, com
    G = ∫∫∫ dνa dνb dωa  ξ(νa)ξ(νb)ξ(ωa)ξ(ωb) correction(ωa) correction(ωb)
          × [-i(χγ²/π) K(E) / (Γ(νa)Γ(νb)Γ(ωa)Γ(ωb))] Σ_j A^{N-j} B^{j-1}
    onde ωb = νa+νb-ωa e E = ωa+ωb = νa+νb.

    Com a(ω) = t(ω) e^{iωL}, tem-se A = a(ωa)a(νb) e B = a(ωb)a(νa), logo o
    termo j fatoriza em [a^{N-j}(ωa) a^{j-1}(ωb)] × [a^{j-1}(νa) a^{N-j}(νb)].
    Como K só depende de E, cada termo vira uma integral 1D em E sobre o
    produto de duas convoluções.
    """
    w, h = grade(sigma, N, L)

    xi, c, G = xi_in(w, sigma), correction(w, N, L), Gamma(w)
    a = t_site(w) * np.exp(1j * w * L)

    E = np.linspace(2 * w[0], 2 * w[-1], 2 * len(w) - 1)   # energia total
    K = 1.0 / (1.0 + 2j * chi / (gamma + 1j * (2 * Delta - E)))

    def conv(p, q):
        return fftconvolve(p, q) * h

    base_out = xi * c / G   # fatores de ωa, ωb (saída)
    base_in = xi / G        # fatores de νa, νb (entrada)

    soma = np.zeros_like(E, dtype=complex)
    for j in range(1, N + 1):
        f, g = a ** (N - j), a ** (j - 1)
        soma += conv(base_out * f, base_out * g) * conv(base_in * g, base_in * f)

    return 1.0 - (1j * chi * gamma ** 2 / np.pi) * simpson(K * soma, x=E)


def fidelity_F1(F, phi_target=None):
    """Fidelidade média de porta F1 (d=4, A=diag(1,1,1,F)).

    Com `phi_target` dá F1(phi) = [6 + 3 Re(e^{i phi} F) + |F|^2]/10; com
    None dá F1(phi_opt) = [6 + 3|F| + |F|^2]/10, que escolhe phi de modo que
    e^{i phi} F = |F|. Vale também para arrays."""
    F_abs = np.abs(F)
    if phi_target is None:
        return (6 + 3 * F_abs + F_abs ** 2) / 10.0
    return (6 + 3 * np.real(np.exp(1j * phi_target) * F) + F_abs ** 2) / 10.0


def run(N):
    sigma_vals = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    L_vals = np.linspace(L_min, L_max, n_L)

    F_grid = np.zeros((len(L_vals), len(sigma_vals)), dtype=complex)
    for i, L in enumerate(L_vals):
        for j, sigma in enumerate(sigma_vals):
            F_grid[i, j] = compute_F(sigma, N, L)
        if not quiet:
            print(f"N={N}  L={L:6.3f}  concluído ({i+1}/{len(L_vals)})")

    F_abs = np.abs(F_grid)
    F1_pi = fidelity_F1(F_grid, phi)
    F1_opt = fidelity_F1(F_grid)
    if np.any(F_abs > 1.0 + 1e-6) and not quiet:
        print("AVISO: |F| > 1 em alguns pontos -- considere aumentar "
              "'pts_per_width' ou 'k_sigma'.")
    if not quiet:
        i, j = np.unravel_index(np.argmax(F1_pi), F1_pi.shape)
        print(f"N={N}  max F1(phi={phi:.3g}) = {F1_pi[i, j]:.6f} em "
              f"sigma={sigma_vals[j]:.4g}, L={L_vals[i]:.4g}")

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile_name(N))
    np.savez(
        outpath,
        sigmas=sigma_vals,
        Ls=L_vals,
        F_abs=F_abs,
        F_re=np.real(F_grid),
        F_im=np.imag(F_grid),
        F1_pi=F1_pi,
        F1_opt=F1_opt,
        N=N, chi=chi, gamma=gamma, Delta=Delta, omega0=omega0, phi=phi,
        k_sigma=k_sigma, pts_per_width=pts_per_width,
    )
    if not quiet:
        print(f"Dados salvos em {outpath}")


def main():
    for N in N_values:
        run(N)


if __name__ == '__main__':
    main()
