# -*- coding: utf-8 -*-
"""
Figura 4: espectro do foton de saida para um sitio de Brod-Combes com o atomo
acoplado a UMA mola (modo de banho) em estado de Fock |n>.
Colunas: mola rapida (esquerda) e mola lenta (direita).
Linhas : foton com cor exata (em cima) e foton com largura sigma (embaixo).
Os graficos de cima usam o MESMO eixo x dos de baixo (sharex por coluna).

Formulas (PDF "Um sitio de Brod-Combes com um banho em estado de Fock"):
  t_mn(nu) = delta_mn - gamma * sum_k F_km F_kn /
             [gamma/2 + i(Delta - 4g^2/omega + (k-n) omega - nu)]
  foton sai em  omega_a = nu_a - (m-n) omega   (conservacao de energia)
  S_out(omega_a) = sum_m |xi(omega_a+(m-n)omega)|^2 |t_mn(omega_a+(m-n)omega)|^2
"""
import numpy as np
from scipy.special import eval_genlaguerre, factorial
import matplotlib.pyplot as plt

# ---------------- parametros do atomo (unidades de gamma) ----------------
gam   = 1.0      # taxa de emissao do atomo no guia
Delta = 0.0      # dessintonia do atomo

# ---------------- parametros do banho e do foton ----------------
n     = 1        # quanta iniciais na mola (n>=1 permite linha "azul")
S     = 0.25     # fator de Huang-Rhys S=(2g/omega)^2, igual nos dois casos
sig   = 0.3     # largura espectral do pacote do foton (graficos de baixo)
K     = 60       # corte da soma sobre os niveis k do oscilador deslocado
dms   = range(-1, 5)   # valores de m-n mostrados (m = n-1, ..., n+3)
casos = [("Mola rapida", 3.0, (-15.5, 3.5)),  # (titulo, omega, limites do eixo x)
         ("Mola lenta",  0.1, (-1.6, 1.6))]

cores = {-1: 'tab:blue', 0: 'k', 1: 'tab:red', 2: 'tab:orange', 3: 'tab:purple', 4: 'tab:green'}
nomes = {-1: 'm=n-1 (ganha energia)', 0: 'm=n (limpo)', 1: 'm=n+1 (perde energia)',
         2: 'm=n+2', 3: 'm=n+3', 4: 'm=n+4'}


def FC(k, n, beta):
    """Fator de Franck-Condon <k|D(beta)|n>, beta real."""
    if k >= n:
        return (np.sqrt(factorial(n) / factorial(k)) * beta**(k - n)
                * np.exp(-beta**2 / 2) * eval_genlaguerre(n, k - n, beta**2))
    return (-1)**(n - k) * FC(n, k, beta)


def t_mn(m, n, nu, w, g):
    """Amplitude de o foton entrar com cor nu e a mola ir de |n> para |m>."""
    beta = 2 * g / w
    s = 0.0
    for k in range(K):
        den = gam / 2 + 1j * (Delta - 4 * g**2 / w + (k - n) * w - nu)
        s += FC(k, m, beta) * FC(k, n, beta) / den
    return (1.0 if m == n else 0.0) - gam * s


fig, ax = plt.subplots(2, 2, figsize=(11, 6.6), sharex='col')

for j, (titulo, w, xlim) in enumerate(casos):
    g = np.sqrt(S) * w / 2                  # acoplamento que da o S escolhido
    nu0 = Delta #- 4 * g**2 / w              # foton centrado no vale do meio
    xi2 = lambda v: np.exp(-(v - nu0)**2 / (2 * sig**2)) / np.sqrt(2 * np.pi * sig**2)

    # ---- em cima: foton monocromatico em nu0 -> linhas ----
    print(titulo, ' nu_a =', round(nu0, 3))
    soma = 0.0
    for d in dms:
        m = n + d
        p = abs(t_mn(m, n, nu0, w, g))**2   # probabilidade da linha
        wa = nu0 - d * w                     # cor de saida dessa linha
        soma += p
        ax[0, j].vlines(wa, 0, p, color=cores[d], lw=3, label=nomes[d])
        print(f'   m-n={d:+d}: omega_a={wa:+.3f}  prob={p:.4f}')
    print('   soma =', round(soma, 5))
    ax[0, j].set_ylim(0, 1.05)
    ax[0, j].set_ylabel('probabilidade')
    ax[0, j].set_title(f'{titulo} ($\\omega={w:g}\\gamma$), $n={n}$: foton com cor exata')

    # ---- embaixo: pacote gaussiano de largura sig ----
    wa = np.linspace(xlim[0], xlim[1], 2001)
    total = np.zeros_like(wa)
    for d in dms:
        m = n + d
        nus = wa + d * w                     # cor de entrada que leva a saida wa
        Sm = xi2(nus) * np.array([abs(t_mn(m, n, v, w, g))**2 for v in nus])
        total += Sm
        ax[1, j].plot(wa, Sm, color=cores[d], lw=1.3)
    ax[1, j].plot(wa, xi2(wa), color='gray', ls=':', lw=1.2, label='espectro de entrada')
    ax[1, j].plot(wa, total, color='tab:green', ls='--', lw=1.5, label='espectro de saida total')
    ax[1, j].set_title(f'{titulo}: foton com largura $\\sigma={sig}\\gamma$')
    ax[1, j].set_xlabel(r'$\omega_a/\gamma$')
    ax[1, j].set_ylabel('densidade espectral')
    ax[1, j].set_xlim(*xlim)                 # mesmo eixo x em cima e embaixo

ax[0, 0].legend(fontsize=7.5, loc='upper left')
ax[1, 1].legend(fontsize=7.5, loc='upper right')
plt.tight_layout()
plt.savefig(rf'figura4_espectro_n{n}_sigma{sig}.svg', dpi=300)
# plt.savefig('figura4_espectro_n{n}.pdf')
plt.show()