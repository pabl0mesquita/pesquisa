# -*- coding: utf-8 -*-
"""
Espectro do foton de saida para N sitios de Brod-Combes em cascata, cada atomo
acoplado ao seu proprio oscilador (mola) em estado de Fock |n>.
Esquerda: mola rapida (omega = 3 gamma).  Direita: mola lenta (omega = 0.1 gamma).
Em cada painel: N = 1, 2, 3, 4.

Formulas (PDF "N sitios de Brod-Combes em cascata ... o elemento de matriz de um foton"):
  <omega_a, m | nu_a, n> = prod_j t^(j)_{m_j n_j}(x_j) delta(omega_a - nu_a + sum_j delta_j)
  delta_j = (m_j - n_j) omega          (energia deixada na mola do sitio j)
  x_j     = nu_a - sum_{l<j} delta_l   (cor com que o foton chega ao sitio j)

  S_out(omega_a) = sum_m |xi(omega_a + D_m)|^2 prod_j |t^(j)_{m_j n}(x_j)|^2,
  D_m = sum_j delta_j,   x_j = omega_a + D_m - sum_{l<j} delta_l .
Sitios identicos: mesmos gamma, Delta, omega, g e n em todos (S = (2g/omega)^2).
"""
import numpy as np
from scipy.special import eval_genlaguerre, factorial
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'svg.fonttype': 'path',
})

# ---------------- parametros do atomo (unidades de gamma) ----------------
gam   = 1.0      # taxa de emissao de cada atomo no guia
Delta = 0.0      # dessintonia dos atomos

# ---------------- parametros do banho e do foton ----------------
n       = 0          # quanta iniciais em cada mola (o mesmo em todos os sitios)
sig     = 0.3        # largura espectral do pacote do foton (desvio padrao de |xi|^2)
omega_c = None      # frequencia central do pacote; None = ressonancia do sistema
                     # completo, Delta - 4g^2/omega
K       = 60         # corte da soma sobre os niveis k do oscilador deslocado
dms     = range(-1, 4)   # valores de m_j - n permitidos em CADA sitio
N_values = [1, 4, 8, 10]  # numeros de sitios (uma curva por N)
# (titulo, omega, g, limites do eixo x); o fator de Huang-Rhys de cada mola
# e calculado a partir de g e omega: S = (2g/omega)^2
casos = [("", 10.0, 1, (-12.5, 5)),
         ("",  0.1, 1, (-1.6, 1.6))]

cores_N = plt.cm.viridis(np.linspace(0, 0.9, len(N_values)))   # uma cor por N, qualquer quantidade


def xi2(nu, w_c):
    """Pacote gaussiano do foton de entrada, |xi(nu)|^2 (normalizado na intensidade)."""
    return np.exp(-(nu - w_c)**2 / (2 * sig**2)) / np.sqrt(2 * np.pi * sig**2)


def FC(k, m, beta):
    """Fator de Franck-Condon <k|D(beta)|m>, beta real."""
    if k >= m:
        return (np.sqrt(factorial(m) / factorial(k)) * beta**(k - m)
                * np.exp(-beta**2 / 2) * eval_genlaguerre(m, k - m, beta**2))
    return (-1)**(m - k) * FC(m, k, beta)


def tabela_t(w, g):
    """Devolve uma funcao t(m, nu) = t_{m n}(nu) vetorizada em nu (mola de frequencia w, acoplamento g)."""
    beta = 2 * g / w
    ms = [n + d for d in dms if n + d >= 0]
    F = {m: np.array([FC(k, m, beta) for k in range(K)]) for m in ms + [n]}
    ks = np.arange(K)

    def t(m, nu):
        nu = np.asarray(nu, float)
        den = gam / 2 + 1j * (Delta - 4 * g**2 / w + (ks[:, None] - n) * w - nu.ravel()[None, :])
        s = (F[m] * F[n]) @ (1.0 / den)
        return ((1.0 if m == n else 0.0) - gam * s).reshape(nu.shape)
    return t, ms


def espectro(N, w, g, wa, w_c):
    """S_out(omega_a) para N sitios identicos.

    Em vez de somar as (len(ms))^N configuracoes (m_1,...,m_N) uma a uma, usamos
    que x_j = omega_a + R_j, com R_j = sum_{l>=j} delta_l (energia deixada do sitio
    j em diante). Andando do ultimo sitio para o primeiro:
        f_{N+1}(R) = 1 se R = 0
        f_j(R)     = sum_{m_j} |t_{m_j n}(omega_a + R)|^2 f_{j+1}(R - delta_{m_j})
        S_out      = sum_R |xi(omega_a + R)|^2 f_1(R)
    R e sempre um multiplo inteiro r de omega, entao o custo cresce so como N^2.
    """
    t, ms = tabela_t(w, g)
    abs_t2 = {}                               # |t_mn(omega_a + r omega)|^2, com cache
    def T2(m, r):
        if (m, r) not in abs_t2:
            abs_t2[(m, r)] = np.abs(t(m, wa + r * w))**2
        return abs_t2[(m, r)]
    f = {0: np.ones_like(wa)}                 # f_{N+1}: so R = 0
    for _ in range(N):                        # sitios N, N-1, ..., 1
        novo = {}
        for r_ant, val in f.items():
            for m in ms:
                r = r_ant + (m - n)           # R_j = delta_j + R_{j+1}
                novo[r] = novo.get(r, 0.0) + T2(m, r) * val
        f = novo
    return sum(xi2(wa + r * w, w_c) * val for r, val in f.items())


fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

for j, (titulo, w, g, xlim) in enumerate(casos):
    S = (2 * g / w)**2                      # fator de Huang-Rhys dependente de g e omega
    w_c = Delta - 4 * g**2 / w if omega_c is None else omega_c
    wa = np.linspace(xlim[0] - 3 * w, xlim[1] + 3 * w, 6001)   # grade um pouco maior para a normalizacao
    for iN, N in enumerate(N_values):
        Sout = espectro(N, w, g, wa, w_c)
        area = np.trapezoid(Sout, wa)
        print(f'{titulo}: N={N}  area do espectro = {area:.5f}')
        ax[j].plot(wa, Sout, color=cores_N[iN], lw=1.3, label=rf'$N={N}$')
    ax[j].plot(wa, xi2(wa, w_c), color='gray', ls=':', lw=1, label=r'$|\xi(\nu)|^2$')
    ax[j].set_xlim(*xlim)
    ax[j].set_title(rf'{titulo} ($\omega={w:g}\gamma$, $g={g:g}\gamma$, $S={S:.3g}$, $n={n}$, $\sigma={sig:g}\gamma$)',
                    fontsize=10)
    ax[j].set_xlabel(r'$\omega_a/\gamma$')
    ax[j].set_ylabel(r'Densidade espectral $S_{\rm out}(\omega_a)$')
    ax[j].legend(fontsize=8, loc='upper left' if j == 0 else 'upper right')

fig.tight_layout()
tag_g = '-'.join(f'{c[2]:g}' for c in casos)   # valores de g usados, no nome do arquivo
fig.savefig(rf'./novos/espectro_N_sitios_n{n}_sigma{sig}_g{tag_g}_ressonancia.svg', dpi=300, bbox_inches='tight')
fig.savefig(rf'./novos/espectro_N_sitios_n{n}_sigma{sig}_g{tag_g}_ressonancia.png', dpi=300, bbox_inches='tight')
plt.show()