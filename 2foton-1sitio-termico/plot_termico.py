# -*- coding: utf-8 -*-
"""
Plota F1(sigma) a partir dos .npz gerados por calcular_termico.py (molas em
estado termico, aproximacao de mola lenta), no estilo da Fig. 5 de Brod & Combes.
Cada entrada de `curvas` e um arquivo (um caso) com sua cor, rotulo e estilo.
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
sufixo = 'chi1000_gam1_w00'   # parametros do sitio usados em calcular_termico.py

# (arquivo, cor, rotulo, estilo)
curvas = [
    (f'termico_{sufixo}_wa0.3_Sa0_nbara0_wb0.3_Sb0_nbarb0_media.npz',     'red',     'sem mola',              '-'),
    (f'termico_{sufixo}_wa0.3_Sa4_nbara0_wb0.3_Sb4_nbarb0_media.npz',     'black',   r'$\bar n=0$',           '--'),
    (f'termico_{sufixo}_wa0.3_Sa4_nbara1_wb0.3_Sb4_nbarb1_media.npz',     '#8A2BE2', r'$\bar n=1$',           '--'),
    (f'termico_{sufixo}_wa0.3_Sa4_nbara2_wb0.3_Sb4_nbarb2_media.npz',   '#1E90FF', r'$\bar n=2$',          '--'),
]

xlim = (0.03, 100)
ylim = (0.35, 1.0)
saida = 'fidelidade_sigma_mola_termico.svg'

fig, ax = plt.subplots(figsize=(7, 4.5))
for arq, cor, rot, style in curvas:
    d = np.load(os.path.join(pasta, arq))
    ax.semilogx(d['sigmas'], d['F1'], style, color=cor, lw=1.2, label=fr'$F_1$ {rot}')
    # ax.semilogx(d['sigmas'], d['F2'], ':', color=cor, lw=1.2, label=fr'$F_2$ {rot}')

ax.set_title(rf"$\chi={float(d['chi']):g},\ \gamma={float(d['gam']):g},\ "
             rf"\omega_0={float(d['omega0']):g},\ \omega^{{(a,b)}}={float(d['w_a']):g},\ "
             rf"S=(2g/\omega)^2={float(d['S_a']):g}$  (mola lenta, térmica)", fontsize=10)
ax.set_xlabel(r'Largura do pacote de onda ($\sigma$)')
ax.set_ylabel(r'Fidelidade ($F$)')
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.legend(fontsize=8, ncol=2, loc='upper right')
fig.tight_layout()
fig.savefig(saida, dpi=300, bbox_inches='tight')
fig.savefig(saida.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
plt.show()
