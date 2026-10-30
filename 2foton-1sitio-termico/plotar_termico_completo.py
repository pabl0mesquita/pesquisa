# -*- coding: utf-8 -*-
"""
Plota F1(sigma) a partir dos .npz gerados por calcular_termico_completo.py
(molas em estado termico, solucao completa, sem aproximacao de mola lenta),
no estilo da Fig. 5 de Brod & Combes.  Opcionalmente sobrepoe, pontilhado,
o resultado da aproximacao de mola lenta (calcular_termico.py) para comparar.
Cada entrada de `curvas` e um caso com sua cor, rotulo e estilo.
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
sufixo = 'chi5_gam4.5_w01.1'   # parametros do sitio usados em calcular_termico_completo.py
comparar_lenta = True          # sobrepoe a aproximacao de mola lenta (se o arquivo existir)

# (caso, cor, rotulo, estilo);  caso = parte do nome do arquivo com as molas
curvas = [
    ('wa1_Sa0_nbara0_wb1_Sb0_nbarb0',     'red',     'sem mola',    '-'),
    ('wa1_Sa0.1_nbara0_wb1_Sb0.1_nbarb0', 'black',   r'$\bar n=0$', '-.'),
    ('wa1_Sa0.1_nbara1_wb1_Sb0.1_nbarb1', '#8A2BE2', r'$\bar n=1$', '-.'),
    ('wa1_Sa0.1_nbara2_wb1_Sb0.1_nbarb2', '#1E90FF', r'$\bar n=2$', '-.'),
]

xlim = (0.03, 100)
ylim = (0.35, 1.0)
saida = 'fidelidade_sigma_mola_termico_completo.svg'

fig, ax = plt.subplots(figsize=(7, 4.5))
for caso, cor, rot, style in curvas:
    d = np.load(os.path.join(pasta, f'termico_completo_{sufixo}_{caso}_media.npz'))
    ax.semilogx(d['sigmas'], d['F1'], style, color=cor, lw=1.2, label=fr'$F_1$ {rot}')
    lenta = os.path.join(pasta, f'termico_{sufixo}_{caso}_media.npz')
    if comparar_lenta and float(d['S_a']) > 0 and os.path.exists(lenta):
        dl = np.load(lenta)
        ax.semilogx(dl['sigmas'], dl['F1'], ':', color=cor, lw=1.0)
if comparar_lenta:
    ax.plot([], [], ':', color='gray', lw=1.0, label='mola lenta')

ax.set_title(rf"$\chi={float(d['chi']):g},\ \gamma={float(d['gam']):g},\ "
             rf"\omega_0={float(d['omega0']):g},\ \omega^{{(a,b)}}={float(d['w_a']):g},\ "
             rf"S=(2g/\omega)^2={float(d['S_a']):g}$  (térmica, completa)", fontsize=10)
ax.set_xlabel(r'Largura do pacote de onda ($\sigma$)')
ax.set_ylabel(r'Fidelidade ($F$)')
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.legend(fontsize=8, ncol=2, loc='upper right')
fig.tight_layout()
fig.savefig(saida, dpi=300, bbox_inches='tight')
fig.savefig(saida.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
plt.show()
