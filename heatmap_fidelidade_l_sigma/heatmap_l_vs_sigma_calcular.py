"""
Calcula o mapa da fidelidade média de porta F1 no plano (sigma, L) para dois
sítios cross-Kerr idênticos, fótons contrapropagantes, e salva os dados em
data/heatmap para serem plotados por heatmap_l_vs_sigma_plotar.py.

A fidelidade é F1(phi) = [6 + 3 Re(e^{i phi} F) + |F|^2] / 10, onde
F = <xi xi| S1†S1† S |xi xi> é a amplitude de sobreposição (|F| é o módulo do
produto escalar, que ignora a fase e por isso não é a fidelidade). São salvos
F1(phi = pi), que é a porta CZ, e F1(phi_opt), com a fase escolhida de modo a
maximizar F1; F, |F|, Re F e Im F seguem salvos para diagnóstico.

Usa as mesmas expressões e o mesmo método de integração de calcular.py: a
integral tripla é reduzida a convoluções 1D (o núcleo fatoriza, com K dependendo
só da energia total E = wa+wb), o que permite resolver sigma e gamma
simultaneamente.
"""
import os

import numpy as np
from scipy.integrate import simpson
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------
# Parâmetros físicos fixos
# ----------------------------------------------------------------------
gamma = 1.0           # gamma_1 = gamma_2 (fixo)
chi = 1000.0          # chi_1 = chi_2 (fixo)
Delta = 0.0           # Delta_1 = Delta_2 (convenção)
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
# Parâmetros numéricos da integração (mesma convenção de calcular.py)
# ----------------------------------------------------------------------
k_sigma = 9.0         # metade da janela de integração, em unidades de sigma
pts_per_width = 14    # pontos por menor escala relevante, min(sigma, gamma)
n_min = 2001          # número mínimo de pontos na grade
n_max = 600001        # teto de pontos (proteção de tempo/memória)

# ----------------------------------------------------------------------
# Parâmetros de saída
# ----------------------------------------------------------------------
outdir = os.path.join('data', 'heatmap')          # pasta onde os dados serão salvos
outfile = 'heatmap_chi1000_gamma1_omega0.npz'     # nome do arquivo .npz de saída
quiet = False         # se True, não imprime o progresso da varredura


def Gamma(w):
    return gamma / 2 + 1j * (Delta - w)


def t_site(w):
    """Fase de espalhamento de um único sítio (|t|=1)."""
    return np.conj(Gamma(w)) / Gamma(w)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def correction(w, L):
    """Remove a amplitude completa de fóton único (dois sítios): conj(t^2),
    incluindo a fase de propagação e^{iwL} entre os sítios. Sem ela a
    referência linear e os termos não lineares ficam em convenções
    diferentes e S deixa de ser unitária para L != 0."""
    return np.conj(t_site(w)) ** 2 * np.exp(-1j * w * L)


def grade(sigma):
    """Grade de frequências e seu espaçamento.

    As quatro gaussianas confinam o integrando a |w-w_c| <~ k_sigma*sigma (o
    núcleo decai como 1/w⁴), então a janela é fixada só por sigma. Já o
    espaçamento precisa resolver a menor escala presente, min(sigma, gamma):
    é o que garante convergência tanto para sigma << gamma quanto sigma >> gamma.
    """
    half = k_sigma * sigma
    n = int(2 * half / (min(sigma, gamma) / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1   # ímpar: Simpson com nº par de intervalos
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


def compute_F(sigma, L):
    """
    F = 1 + G  (produto interno <target|espalhado>, com a deformação de fóton
    único já removida por `correction`), com

    G = ∫∫∫ dwa dwb dna  xi(wa)xi(wb)xi(na)xi(nb) *
                          correction(wa) correction(wb) *
                          [termo 2 + termo 3](wa,wb,na,nb)
    onde nb = wa+wb-na (vem da delta).

    Com E = wa+wb = na+nb, K depende só de E e cada fator restante depende de
    uma única frequência: a integral tripla vira uma integral 1D em E sobre o
    produto de duas convoluções.
    """
    w, h = grade(sigma)

    xi = xi_in(w, sigma)
    c = correction(w, L)
    eL = np.exp(1j * w * L)
    G = Gamma(w)
    t = t_site(w)

    E = np.linspace(2 * w[0], 2 * w[-1], 2 * len(w) - 1)   # energia total
    K = 1.0 / (1.0 + 2j * chi / (gamma + 1j * (2 * Delta - E)))

    def conv(a, b):
        return fftconvolve(a, b) * h

    # Termo 2 (não linearidade no sítio 1). Com sítios idênticos o termo 3
    # (não linearidade no sítio 2) dá exatamente o mesmo valor — as duas
    # convoluções apenas trocam de ordem —, daí o fator 2.
    termo = -(1j * chi * gamma ** 2 / np.pi) * simpson(
        K * conv(xi * c * eL * t / G, xi * c / G)
          * conv(xi / G, xi * eL * t / G), x=E)

    return 1.0 + 2 * termo


def fidelity_F1(F, phi_target=None):
    """Fidelidade média de porta F1 (d=4, A=diag(1,1,1,F)).

    Com `phi_target` dá F1(phi) = [6 + 3 Re(e^{i phi} F) + |F|^2]/10; com
    None dá F1(phi_opt) = [6 + 3|F| + |F|^2]/10, que escolhe phi de modo que
    e^{i phi} F = |F|. Vale também para arrays."""
    F_abs = np.abs(F)
    if phi_target is None:
        return (6 + 3 * F_abs + F_abs ** 2) / 10.0
    return (6 + 3 * np.real(np.exp(1j * phi_target) * F) + F_abs ** 2) / 10.0


def main():
    sigma_vals = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)
    L_vals = np.linspace(L_min, L_max, n_L)

    F_grid = np.zeros((len(L_vals), len(sigma_vals)), dtype=complex)
    for i, L in enumerate(L_vals):
        for j, sigma in enumerate(sigma_vals):
            F_grid[i, j] = compute_F(sigma, L)
        if not quiet:
            print(f"L={L:6.3f}  concluído ({i+1}/{len(L_vals)})")

    F_abs = np.abs(F_grid)
    F1_pi = fidelity_F1(F_grid, phi)
    F1_opt = fidelity_F1(F_grid)
    if np.any(F_abs > 1.0 + 1e-6) and not quiet:
        print("AVISO: |F| > 1 em alguns pontos -- considere aumentar "
              "'pts_per_width' ou 'k_sigma'.")
    if not quiet:
        i, j = np.unravel_index(np.argmax(F1_pi), F1_pi.shape)
        print(f"max F1(phi={phi:.3g}) = {F1_pi[i, j]:.6f} em sigma={sigma_vals[j]:.4g}, L={L_vals[i]:.4g}")

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile)
    np.savez(
        outpath,
        sigmas=sigma_vals,
        Ls=L_vals,
        F_abs=F_abs,
        F_re=np.real(F_grid),
        F_im=np.imag(F_grid),
        F1_pi=F1_pi,
        F1_opt=F1_opt,
        chi=chi, gamma=gamma, Delta=Delta, omega0=omega0, phi=phi,
        k_sigma=k_sigma, pts_per_width=pts_per_width,
    )
    if not quiet:
        print(f"Dados salvos em {outpath}")


if __name__ == '__main__':
    main()
