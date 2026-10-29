# -*- coding: utf-8 -*-
"""
Figura 1 (lado a lado): chance de o foton sair "limpo", |t_nn(nu_a)|^2,
para um sitio de Brod-Combes com o atomo acoplado a UM modo de banho
("mola") em estado de Fock |n>.  Esquerda: mola rapida.  Direita: mola lenta.

Formula usada (PDF "Um sitio de Brod-Combes com um banho em estado de Fock"):

  t_mn(nu_a) = delta_mn - gamma * sum_k  F_km F_kn /
               [ gamma/2 + i( Delta - 4 g^2/omega + (k-n) omega - nu_a ) ]

  F_kn = <k|D(beta)|n>   (fator de Franck-Condon),  beta = 2 g / omega
"""
import numpy as np
from scipy.special import eval_genlaguerre, factorial
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Parametros do atomo (todas as frequencias em unidades de gamma)
# ------------------------------------------------------------------
gam   = 1.0     # gamma: taxa de emissao do atomo no guia (define a unidade)
Delta = 0.0     # Delta: dessintonia do atomo (onde fica a linha sem banho)

# ------------------------------------------------------------------
# Parametros das duas molas (banhos)
# S = (2g/omega)^2 e o fator de Huang-Rhys ("tamanho do puxao").
# Mantemos o MESMO S nos dois casos e mudamos so a velocidade omega,
# para comparar apenas o efeito de a mola ser rapida ou lenta.
# ------------------------------------------------------------------
S = 0.25                                    # fator de Huang-Rhys
casos = [
    # (titulo,        omega, K)
    ("Mola rápida",   3.0,   60),            # omega >> gamma: bandas laterais separadas
    ("Mola lenta",    0.3,   40),            # omega << gamma: bandas dentro da linha
]
# K: numero de niveis k do oscilador deslocado mantidos na soma (corte numerico)

ns    = [0, 1, 2]                           # estados de Fock iniciais da mola (qualquer lista)
nus   = np.linspace(-5, 5, 801)             # eixo x: cor do foton que entra, nu_a/gamma

# ------------------------------------------------------------------
# Estilo de cada curva (color: cor, ls: tipo de linha, lw: espessura)
# ls: '-' cheia, '--' tracejada, ':' pontilhada, '-.' traco-ponto
# ------------------------------------------------------------------
estilos = [                                 # curvas |t_nn|^2: estilos[i] vale para ns[i]
    dict(color='k', ls='--', lw=1),
    dict(color='b', ls='--', lw=1),
    dict(color='r', ls='--', lw=1),
]
assert len(estilos) >= len(ns), 'defina um estilo em "estilos" para cada n de "ns"'
estilo_sem_banho  = dict(color='purple', ls='-.', lw=1.0)  # caso g = 0
estilo_vales      = dict(color='gray',   ls=':',  lw=0.5)  # linhas verticais dos vales
estilo_vales_leg  = dict(color='gray',   ls=':',  lw=1.0)  # mesma, na legenda (mais grossa)


# ------------------------------------------------------------------
# Fator de Franck-Condon <k|D(beta)|n> para beta real
# ------------------------------------------------------------------
def FC(k, n, beta):
    if k >= n:
        return (np.sqrt(factorial(n) / factorial(k)) * beta**(k - n)
                * np.exp(-beta**2 / 2) * eval_genlaguerre(n, k - n, beta**2))
    return (-1)**(n - k) * FC(n, k, beta)   # simetria para k < n


# ------------------------------------------------------------------
# Amplitude t_mn(nu_a): mola vai de |n> para |m>
# ------------------------------------------------------------------
def t_mn(m, n, nu, gam, Delta, w, g, K):
    beta = 2 * g / w
    s = 0.0
    for k in range(K):
        den = gam / 2 + 1j * (Delta - 4 * g**2 / w + (k - n) * w - nu)
        s += FC(k, m, beta) * FC(k, n, beta) / den
    return (1.0 if m == n else 0.0) - gam * s


# ------------------------------------------------------------------
# Figura
# ------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for ax, (titulo, w, K) in zip(axs, casos):
    g = np.sqrt(S) * w / 2                  # acoplamento que da o S escolhido
    zpl = Delta - 4 * g**2 / w              # vale do meio (deslocado pelo polaron)

    # caso sem banho (g = 0): t = 1 - gamma / [gamma/2 + i(Delta - nu_a)]
    sem_banho = np.abs(1 - gam / (gam / 2 + 1j * (Delta - nus)))**2
    ax.plot(nus, sem_banho, **estilo_sem_banho, label='sem banho')

    # corte K: a soma em k precisa cobrir toda a distribuicao F_kn^2 de cada n
    for n in ns:
        norma = sum(FC(k, n, 2 * g / w)**2 for k in range(K))
        if norma < 0.999:
            print(f'AVISO {titulo}: n={n}, sum_k F_kn^2 = {norma:.4f} -> aumente K')

    for n, estilo in zip(ns, estilos):
        limpo = [abs(t_mn(n, n, nu, gam, Delta, w, g, K))**2 for nu in nus]
        ax.plot(nus, limpo, **estilo, label=f'$n={n}$')
        print(f'{titulo}: n={n}, minimo de |t_nn|^2 = {min(limpo):.3f}')

    # linhas pontilhadas: so as posicoes de vale nu_a = Delta - 4g^2/w + j*omega
    # cujo peso de Franck-Condon F_{n+j,n}^2 passa de 1% para algum n plotado
    beta = 2 * g / w
    for j in range(-max(ns), K - max(ns)):
        peso = max(FC(n + j, n, beta)**2 for n in ns if n + j >= 0)
        x = zpl + j * w
        if peso > 0.01 and -5 <= x <= 5:
            ax.axvline(x, **estilo_vales)
    # entrada da legenda para as linhas pontilhadas (traco mais grosso para ficar visivel)
    ax.plot([], [], **estilo_vales_leg,
            label=r'$\nu_a=\Delta-4g^2/\omega+(k-n)\,\omega$')

    ax.set_title(fr'{titulo}: $\omega={w:g}\gamma$, $g={g:.3f}\gamma$, $S={S}$')
    ax.set_xlabel(r'$\nu_a/\gamma$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=9, loc='lower right')

axs[0].set_ylabel(r'$|t_{nn}(\nu_a)|^2$')
plt.tight_layout()
plt.savefig('figura1_rapida_lenta_teste.svg', dpi=300)
plt.savefig('figura1_rapida_lenta_teste.pdf')

plt.show()