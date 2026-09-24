"""
Calcula F1(sigma) para o caso de dois sítios contrapropagante
(Brod & Combes, Fig. 5) e salva os dados em data/counterprop
para serem plotados por plot.py.
"""
import os

import numpy as np
from scipy.integrate import simpson
from scipy.signal import fftconvolve

# ----------------------------------------------------------------------
# Parâmetros físicos (todos configuráveis)
# ----------------------------------------------------------------------
chi1 = 1000            # acoplamento não-linear do sítio 1
chi2 = 1000             # acoplamento não-linear do sítio 2
gamma1 = 1         # taxa de decaimento do sítio 1
gamma2 = 1          # taxa de decaimento do sítio 2
Delta1 = 0.0          # dessintonia do sítio 1
Delta2 = 0.0          # dessintonia do sítio 2
omega0 = 0          # frequência central do pacote, relativa a Delta1
L = 0               # separação entre os sítios (fase de propagação e^{i(...)L})
phi = np.pi           # fase usada no cálculo de F1(phi)

omega_c = Delta1 + omega0  # frequência central efetiva (derivada, não editar)

# ----------------------------------------------------------------------
# Parâmetros da varredura em sigma
# ----------------------------------------------------------------------
sigma_min = 0.01       # menor largura do pacote de onda na varredura
sigma_max = 100.0     # maior largura do pacote de onda na varredura
n_sigma = 100         # número de pontos na varredura de sigma (escala log)

# ----------------------------------------------------------------------
# Parâmetros numéricos da integração
# ----------------------------------------------------------------------
k_sigma = 9.0         # metade da janela de integração, em unidades de sigma
pts_per_width = 14    # pontos por menor escala relevante, min(sigma, gamma)
n_min = 2001          # número mínimo de pontos na grade
n_max = 600001        # teto de pontos (proteção de tempo/memória)

# ----------------------------------------------------------------------
# Parâmetros de saída
# ----------------------------------------------------------------------
outdir = os.path.join('data', 'counterprop')  # pasta onde os dados serão salvos
outfile = 'teste_fidelity_counterprop_l0_chi1000_gamma1_omega0_min001.npz'          # nome do arquivo .npz de saída
quiet = False         # se True, não imprime o progresso da varredura


def Gamma1(w): return gamma1 / 2 + 1j * (Delta1 - w)
def Gamma2(w): return gamma2 / 2 + 1j * (Delta2 - w)


def xi_in(w, sigma):
    """Pacote gaussiano normalizado, Eq. (4)."""
    return (1.0 / (2 * np.pi * sigma ** 2)) ** 0.25 * np.exp(-(w - omega_c) ** 2 / (4 * sigma ** 2))


def phase1(w):
    """Fase de fóton único de dois sítios: Γ1*Γ2*/(Γ1 Γ2). |phase1|=1."""
    return (np.conj(Gamma1(w)) * np.conj(Gamma2(w))) / (Gamma1(w) * Gamma2(w))


def correction(w):
    """Remove a amplitude completa de fóton único: S -> S1†(ωa)S1†(ωb) S.

    Inclui a fase de propagação e^{iwL} entre os sítios, presente nas fases de
    caminho de term2/term3. Sem ela a referência linear e os termos não lineares
    ficam em convenções diferentes e S deixa de ser unitária para L != 0."""
    return np.conj(phase1(w)) * np.exp(-1j * w * L)


def K1(oa, ob):
    return 1.0 / (1.0 + 2j * chi1 / (Gamma1(oa) + Gamma1(ob)))


def K2(oa, ob):
    return 1.0 / (1.0 + 2j * chi2 / (Gamma2(oa) + Gamma2(ob)))


def term2_bracket(oa, ob, na, nb):
    """Termo 2 da expressão de S (sem a delta), com a fase e^{i(oa+nb)L}."""
    phase_L = np.exp(1j * (oa + nb) * L)
    return -phase_L * (1j * chi1 * gamma1 ** 2 / np.pi) \
        * (np.conj(Gamma2(oa)) * np.conj(Gamma2(nb)) / (Gamma2(oa) * Gamma2(nb))) \
        * K1(oa, ob) \
        / (Gamma1(oa) * Gamma1(ob) * Gamma1(na) * Gamma1(nb))


def term3_bracket(oa, ob, na, nb):
    """Termo 3 da expressão de S (sem a delta), com a fase e^{i(na+ob)L}."""
    phase_L = np.exp(1j * (na + ob) * L)
    return -phase_L * (1j * chi2 * gamma2 ** 2 / np.pi) \
        * (np.conj(Gamma1(na)) * np.conj(Gamma1(ob)) / (Gamma1(na) * Gamma1(ob))) \
        * K2(oa, ob) \
        / (Gamma2(oa) * Gamma2(ob) * Gamma2(na) * Gamma2(nb))


# ----------------------------------------------------------------------
# Integral (a delta δ(ωa+ωb-νa-νb) elimina ωb = νa+νb-ωa)
# ----------------------------------------------------------------------
def grade(sigma):
    """Grade de frequências e seu espaçamento.

    As quatro gaussianas confinam o integrando a |ω-ω_c| <~ k_sigma*sigma (o
    núcleo decai como 1/ω⁴), então a janela é fixada só por sigma. Já o
    espaçamento precisa resolver a menor escala presente, min(sigma, gamma):
    é o que garante convergência tanto para sigma << gamma quanto sigma >> gamma.
    """
    half = k_sigma * sigma
    n = int(2 * half / (min(sigma, gamma1, gamma2) / pts_per_width)) + 1
    n = min(max(n, n_min), n_max) | 1   # ímpar: Simpson com nº par de intervalos
    w = np.linspace(omega_c - half, omega_c + half, n)
    return w, w[1] - w[0]


def compute_G(sigma):
    """
    F = 1 + G, com
    G = ∫∫∫ dνa dνb dωa  ξ(νa)ξ(νb)ξ(ωa)ξ(ωb) *
                          correction(ωa) correction(ωb) *
                          [term2_bracket + term3_bracket](ωa,ωb,νa,νb)
    onde ωb = νa+νb-ωa (vem da delta).

    O núcleo fatoriza: com E = νa+νb = ωa+ωb, K1 e K2 dependem só de E
    (Γ(ωa)+Γ(ωb) = γ + i(2Δ - E)) e cada fator restante depende de uma única
    frequência. A integral tripla vira então uma integral 1D em E sobre o
    produto de duas convoluções, o que permite grades finas o bastante para
    resolver sigma e gamma ao mesmo tempo.
    """
    w, h = grade(sigma)

    xi, c = xi_in(w, sigma), correction(w)
    eL = np.exp(1j * w * L)
    G1, G2 = Gamma1(w), Gamma2(w)
    t1, t2 = np.conj(G1) / G1, np.conj(G2) / G2   # fases de fóton único

    E = np.linspace(2 * w[0], 2 * w[-1], 2 * len(w) - 1)   # energia total
    k1 = 1.0 / (1.0 + 2j * chi1 / (gamma1 + 1j * (2 * Delta1 - E)))
    k2 = 1.0 / (1.0 + 2j * chi2 / (gamma2 + 1j * (2 * Delta2 - E)))

    def conv(a, b):
        return fftconvolve(a, b) * h

    # termo 2: não linearidade no sítio 1 (ωa, ωb, νa, νb nessa ordem)
    G2_termo = -(1j * chi1 * gamma1 ** 2 / np.pi) * simpson(
        k1 * conv(xi * c * eL * t2 / G1, xi * c / G1)
           * conv(xi / G1, xi * eL * t2 / G1), x=E)

    # termo 3: não linearidade no sítio 2
    G3_termo = -(1j * chi2 * gamma2 ** 2 / np.pi) * simpson(
        k2 * conv(xi * c / G2, xi * c * eL * t1 / G2)
           * conv(xi * eL * t1 / G2, xi / G2), x=E)

    return G2_termo + G3_termo


def F1_fidelity(sigma):
    G = compute_G(sigma)
    F = 1.0 + G

    if np.abs(F) > 1.0 + 1e-6 and not quiet:
        print(f"AVISO: |F|={np.abs(F):.4f} > 1 em sigma={sigma:.3f} "
              f"(considere aumentar 'pts_per_width' ou 'k_sigma')")

    F1_phi = (6 + 3 * np.real(np.exp(1j * phi) * F) + np.abs(F) ** 2) / 10.0
    F1_opt = (6 + 3 * np.abs(F) + np.abs(F) ** 2) / 10.0
    return F1_phi, F1_opt, F


def main():
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_sigma)

    F1_pi_list, F1_opt_list, F_abs_list, F_re_list, F_im_list = [], [], [], [], []
    for s in sigmas:
        f1_pi, f1_opt, F = F1_fidelity(s)
        F1_pi_list.append(f1_pi)
        F1_opt_list.append(f1_opt)
        F_abs_list.append(np.abs(F))
        F_re_list.append(np.real(F))
        F_im_list.append(np.imag(F))
        if not quiet:
            print(f"sigma={s:8.3f}   |F|={np.abs(F):.4f}   "
                  f"F1(pi)={f1_pi:.4f}   F1(opt)={f1_opt:.4f}")

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile)
    np.savez(
        outpath,
        sigmas=sigmas,
        F1_pi=np.array(F1_pi_list),
        F1_opt=np.array(F1_opt_list),
        F_abs=np.array(F_abs_list),
        F_re=np.array(F_re_list),
        F_im=np.array(F_im_list),
        chi1=chi1, chi2=chi2,
        gamma1=gamma1, gamma2=gamma2,
        Delta1=Delta1, Delta2=Delta2,
        omega0=omega0, L=L, phi=phi,
        k_sigma=k_sigma, pts_per_width=pts_per_width,
        n_grade=len(grade(sigmas[-1])[0]),
    )
    if not quiet:
        print(f"Dados salvos em {outpath}")


if __name__ == '__main__':
    main()
