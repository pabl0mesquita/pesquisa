"""
Calcula F1(sigma) para N sítios idênticos contrapropagantes e salva os dados
em data/n_sitios para serem plotados por plot_sigma_n_sitios.py.

Generaliza calcular.py (N=2) usando a matriz S de N sítios:

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
N = 4                 # número de sítios
chi = 1000            # acoplamento não-linear (igual em todos os sítios)
gamma = 1             # taxa de decaimento (igual em todos os sítios)
Delta = 0.0           # dessintonia (igual em todos os sítios)
omega0 = 0            # frequência central do pacote, relativa a Delta
L_values = [0, 0.4, 0.6, 0.8, 1.0, 2.0]   # separações entre sítios (um .npz por valor)
phi = np.pi           # fase usada no cálculo de F1(phi)

omega_c = Delta + omega0  # frequência central efetiva (derivada, não editar)

# ----------------------------------------------------------------------
# Parâmetros da varredura em sigma
# ----------------------------------------------------------------------
sigma_min = 0.01      # menor largura do pacote de onda na varredura
sigma_max = 100.0     # maior largura do pacote de onda na varredura
n_sigma = 100        # número de pontos na varredura de sigma (escala log)

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


def outfile_name(L):
    """Nome do .npz de saída para uma dada separação L."""
    return f'fidelity_N{N}_L{L:g}_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}.npz'


def Gamma(w): return gamma / 2 + 1j * (Delta - w)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def t(w):
    """Fase de fóton único de um sítio: Γ*/Γ. |t|=1."""
    return np.conj(Gamma(w)) / Gamma(w)


def correction(w, L):
    """Remove a amplitude linear de fóton único <ω-|ω+> = e^{i(N-1)ωL} t(ω)^N."""
    return np.conj(t(w)) ** N * np.exp(-1j * (N - 1) * w * L)


# ----------------------------------------------------------------------
# Integral (a delta δ(ωa+ωb-νa-νb) elimina ωb = νa+νb-ωa)
# ----------------------------------------------------------------------
def grade(sigma, L):
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


def compute_G(sigma, L):
    """
    F = 1 + G, com
    G = ∫∫∫ dνa dνb dωa  ξ(νa)ξ(νb)ξ(ωa)ξ(ωb) correction(ωa) correction(ωb)
          × [-i(χγ²/π) K(E) / (Γ(νa)Γ(νb)Γ(ωa)Γ(ωb))] Σ_j A^{N-j} B^{j-1}
    onde ωb = νa+νb-ωa e E = ωa+ωb = νa+νb.

    Com a(ω) = t(ω) e^{iωL}, tem-se A = a(ωa)a(νb) e B = a(ωb)a(νa), logo o
    termo j fatoriza em [a^{N-j}(ωa) a^{j-1}(ωb)] × [a^{j-1}(νa) a^{N-j}(νb)].
    Como K só depende de E, cada termo vira uma integral 1D em E sobre o
    produto de duas convoluções — a mesma redução de calcular.py (N=2).
    """
    w, h = grade(sigma, L)

    xi, c, G = xi_in(w, sigma), correction(w, L), Gamma(w)
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


def F1_fidelity(sigma, L):
    G = compute_G(sigma, L)
    F = 1.0 + G

    if np.abs(F) > 1.0 + 1e-6 and not quiet:
        print(f"AVISO: |F|={np.abs(F):.4f} > 1 em sigma={sigma:.3f}, L={L:g} "
              f"(considere aumentar 'pts_per_width' ou 'k_sigma')")

    F1_phi = (6 + 3 * np.real(np.exp(1j * phi) * F) + np.abs(F) ** 2) / 10.0
    F1_opt = (6 + 3 * np.abs(F) + np.abs(F) ** 2) / 10.0
    return F1_phi, F1_opt, F


def run(L):
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)

    F1_pi_list, F1_opt_list, F_list = [], [], []
    for s in sigmas:
        f1_pi, f1_opt, F = F1_fidelity(s, L)
        F1_pi_list.append(f1_pi)
        F1_opt_list.append(f1_opt)
        F_list.append(F)
        if not quiet:
            print(f"N={N}  L={L:<5g} sigma={s:8.3f}   |F|={np.abs(F):.4f}   "
                  f"F1(pi)={f1_pi:.4f}   F1(opt)={f1_opt:.4f}")

    F_arr = np.array(F_list)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile_name(L))
    np.savez(
        outpath,
        sigmas=sigmas,
        F1_pi=np.array(F1_pi_list),
        F1_opt=np.array(F1_opt_list),
        F_abs=np.abs(F_arr),
        F_re=np.real(F_arr),
        F_im=np.imag(F_arr),
        N=N, chi=chi, gamma=gamma, Delta=Delta,
        omega0=omega0, L=L, phi=phi,
        k_sigma=k_sigma, pts_per_width=pts_per_width,
        n_grade=len(grade(sigmas[-1], L)[0]),
    )
    if not quiet:
        print(f"Dados salvos em {outpath}")


def main():
    for L in L_values:
        run(L)


if __name__ == '__main__':
    main()
