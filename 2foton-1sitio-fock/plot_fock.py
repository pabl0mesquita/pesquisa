# -*- coding: utf-8 -*-
"""
Plota F1(sigma) (linha cheia) e F2(sigma) (tracejada) a partir dos .npz
gerados por calcular.py, no estilo da Fig. 5 de Brod & Combes.
Cada entrada de `curvas` e um arquivo (um caso) com sua cor e rotulo.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'svg.fonttype': 'path',
})

pasta = 'data'
sufixo = 'chi5_gam4.5_w01.1'   # parametros do sitio usados em calcular.py

# (arquivo, cor, rotulo)
curvas = [
    (f'fid_{sufixo}_wa1_Sa0_na0_wb1_Sb0_nb0_media.npz',     'red',    'sem mola', '-'),
    (f'fid_{sufixo}_wa1_Sa0.25_na0_wb1_Sb0.25_nb0_media.npz', 'black',  r'$n_a=n_b=0$','-.'),
    (f'fid_{sufixo}_wa1_Sa0.25_na1_wb1_Sb0.25_nb1_media.npz', '#8A2BE2', r'$n_a=n_b=1$','-.'),
    (f'fid_{sufixo}_wa1_Sa0.25_na2_wb1_Sb0.25_nb2_media.npz', '#1E90FF', r'$n_a=n_b=2$','-.'),
]

xlim = (0.03, 100)
ylim = (0.35, 1.0)
saida = 'fidelidade_sigma_mola_fock.svg'

fig, ax = plt.subplots(figsize=(7, 4.5))
for arq, cor, rot, style in curvas:
    d = np.load(os.path.join(pasta, arq))
    ax.semilogx(d['sigmas'], d['F1'], style, color=cor, lw=1.2, label=fr'$F_1$ {rot}')
    # ax.semilogx(d['sigmas'], d['F2'], '--', color=cor, lw=1.2, label=fr'$F_2$ {rot}')

ax.set_title(rf"$\chi={float(d['chi']):g},\ \gamma={float(d['gam']):g},\ "
             rf"\omega_0={float(d['omega0']):g},\ \omega^{{(a,b)}}={float(d['w_a']):g},\ "
             rf"S=(2g/\omega)^2={float(d['S_a']):g}$", fontsize=10)
ax.set_xlabel(r'Largura do pacote de onda ($\sigma$)')
ax.set_ylabel(r'Fidelidade ($F$)')
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.legend(fontsize=8, ncol=2, loc='upper right')
fig.tight_layout()
fig.savefig(saida, dpi=300, bbox_inches='tight')
fig.savefig(saida.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
plt.show()
