"""
Calcula F1(L) para N sítios idênticos contrapropagantes, para larguras de
pacote sigma fixas, e salva os dados em data/n_sitios para serem plotados
por plot_l_n_sitios.py.

Mesma física de fidelidade_sigma/calcular_sigma_n_sitios.py, com a varredura
invertida: L varia continuamente e sigma assume alguns valores fixos.

    S = <ωa-|νa+><ωb-|νb+>
        - i(χγ²/π) K(νa,νb) δ(ωa+ωb-νa-νb) / [Γ(νb)Γ(νa)Γ(ωb)Γ(ωa)]
          × Σ_{j=1}^{N} A^{N-j} B^{j-1},

    A = Γ*(ωa)Γ*(νb)/[Γ(ωa)Γ(νb)] e^{i(ωa+νb)L}
    B = Γ*(ωb)Γ*(νa)/[Γ(ωb)Γ(νa)] e^{i(ωb+νa)L}
    <ω-|ν+> = e^{i(N-1)ωL} (Γ*(ω)/Γ(ω))^N δ(ω-ν)
"""
import os

import numpy as np
from scipy.integrate import simpson
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------
# Parâmetros físicos (todos configuráveis)
# ----------------------------------------------------------------------
N_values = [2]        # números de sítios (um .npz por par (N, sigma))
chi = 1000            # acoplamento não-linear (igual em todos os sítios)
gamma = 1             # taxa de decaimento (igual em todos os sítios)
Delta = 0.0           # dessintonia (igual em todos os sítios)
omega0 = 0            # frequência central do pacote, relativa a Delta
sigma_values = [0.184, 0.158, 1, 10]   # larguras do pacote de onda (um .npz por valor)
phi = np.pi           # fase usada no cálculo de F1(phi)

omega_c = Delta + omega0  # frequência central efetiva (derivada, não editar)

# ----------------------------------------------------------------------
# Parâmetros da varredura em L
# ----------------------------------------------------------------------
L_min = 0.0           # menor separação entre sítios na varredura
L_max = 2.0           # maior separação entre sítios na varredura
n_L = 200             # número de pontos na varredura de L (escala linear)

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
outdir = os.path.join('data', 'n_sitios')  # pasta onde os dados serão salvos
quiet = False         # se True, não imprime o progresso da varredura


def outfile_name(N, sigma):
    """Nome do .npz de saída para um dado par (N, sigma)."""
    return f'fidelity_N{N}_sigma{sigma:g}_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}.npz'


def Gamma(w): return gamma / 2 + 1j * (Delta - w)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def t(w):
    """Fase de fóton único de um sítio: Γ*/Γ. |t|=1."""
    return np.conj(Gamma(w)) / Gamma(w)


def correction(w, N, L):
    """Remove a amplitude linear de fóton único <ω-|ω+> = e^{i(N-1)ωL} t(ω)^N."""
    return np.conj(t(w)) ** N * np.exp(-1j * (N - 1) * w * L)


# ----------------------------------------------------------------------
# Integral (a delta δ(ωa+ωb-νa-νb) elimina ωb = νa+νb-ωa)
# ----------------------------------------------------------------------
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


def compute_G(sigma, N, L):
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
    a = t(w) * np.exp(1j * w * L)

    E = np.linspace(2 * w[0], 2 * w[-1], 2 * len(w) - 1)   # energia total
    k = 1.0 / (1.0 + 2j * chi / (gamma + 1j * (2 * Delta - E)))

    def conv(p, q):
        return fftconvolve(p, q) * h

    base_out = xi * c / G   # fatores de ωa, ωb (saída)
    base_in = xi / G        # fatores de νa, νb (entrada)

    soma = np.zeros_like(E, dtype=complex)
    for j in range(1, N + 1):
        f, g = a ** (N - j), a ** (j - 1)
        soma += conv(base_out * f, base_out * g) * conv(base_in * g, base_in * f)

    return -(1j * chi * gamma ** 2 / np.pi) * simpson(k * soma, x=E)


def F1_fidelity(sigma, N, L):
    G = compute_G(sigma, N, L)
    F = 1.0 + G

    if np.abs(F) > 1.0 + 1e-6 and not quiet:
        print(f"AVISO: |F|={np.abs(F):.4f} > 1 em sigma={sigma:g}, N={N}, L={L:.3f} "
              f"(considere aumentar 'pts_per_width' ou 'k_sigma')")

    F1_phi = (6 + 3 * np.real(np.exp(1j * phi) * F) + np.abs(F) ** 2) / 10.0
    F1_opt = (6 + 3 * np.abs(F) + np.abs(F) ** 2) / 10.0
    return F1_phi, F1_opt, F


def run(N, sigma):
    Ls = np.linspace(L_min, L_max, n_L)

    F1_pi_list, F1_opt_list, F_list = [], [], []
    for L in Ls:
        f1_pi, f1_opt, F = F1_fidelity(sigma, N, L)
        F1_pi_list.append(f1_pi)
        F1_opt_list.append(f1_opt)
        F_list.append(F)
        if not quiet:
            print(f"N={N}  sigma={sigma:<5g} L={L:8.3f}   |F|={np.abs(F):.4f}   "
                  f"F1(pi)={f1_pi:.4f}   F1(opt)={f1_opt:.4f}")

    F_arr = np.array(F_list)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile_name(N, sigma))
    np.savez(
        outpath,
        Ls=Ls,
        F1_pi=np.array(F1_pi_list),
        F1_opt=np.array(F1_opt_list),
        F_abs=np.abs(F_arr),
        F_re=np.real(F_arr),
        F_im=np.imag(F_arr),
        N=N, chi=chi, gamma=gamma, Delta=Delta,
        omega0=omega0, sigma=sigma, phi=phi,
        k_sigma=k_sigma, pts_per_width=pts_per_width,
        n_grade=len(grade(sigma, N, Ls[-1])[0]),
    )
    if not quiet:
        print(f"Dados salvos em {outpath}")


def main():
    for N in N_values:
        for sigma in sigma_values:
            run(N, sigma)


if __name__ == '__main__':
    main()
