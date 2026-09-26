"""
Plota o heatmap no plano (sigma, L) a partir dos dados gerados por
heatmap_l_vs_sigma_calcular.py em data/heatmap.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parâmetros de entrada/saída
# ----------------------------------------------------------------------
infile = os.path.join('data', 'heatmap', 'heatmap_chi1000_gamma1_omega0.npz')
dpi = 300             # resolução da imagem salva
show = True           # se True, abre a janela interativa além de salvar

# ----------------------------------------------------------------------
# O que colorir
# ----------------------------------------------------------------------
field = 'F1_pi'       # 'F1_pi' (fidelidade CZ), 'F1_opt' (fidelidade com a melhor fase)
                      # ou 'F_abs' (módulo do produto escalar, não é fidelidade)
vmin, vmax = None, None  # faixa da barra de cores; se None, usa o padrão de `field`
cbar_label = None     # rótulo da barra; se None, usa o padrão de `field`

# arquivo de imagem de saída: um por campo, para não sobrescrever os outros mapas
outfile = f'heatmap_{field}_chi1000_gamma1_omega0_thermal.svg'

# ----------------------------------------------------------------------
# Parâmetros visuais
# ----------------------------------------------------------------------
figsize = (8, 6)      # tamanho da figura (largura, altura) em polegadas
cmap = 'viridis'      # mapa de cores
xlabel = r'$\sigma$'
ylabel = r'$L$'
title = None          # título; se None, é gerado a partir dos parâmetros salvos

FIELD_LABELS = {
    'F_abs': r'$|\langle \mathrm{target}|\phi_{\mathrm{espalhado}}\rangle|$',
    'F1_pi': r'Fidelidade $F_1(\phi=\pi)$',
    'F1_opt': r'Fidelidade $F_1(\phi_{opt})$',
}

# Faixa padrão da barra de cores. A fidelidade parte de 0,4 (sem interação
# efetiva, F = 1) e chega a 1 (porta ideal).
FIELD_RANGES = {
    'F_abs': (0.0, 1.0),
    'F1_pi': (0.4, 1.0),
    'F1_opt': (0.4, 1.0),
}


def default_title(data):
    chi = float(data['chi']) if 'chi' in data else None
    gamma = float(data['gamma']) if 'gamma' in data else None
    omega0 = float(data['omega0']) if 'omega0' in data else None
    if None in (chi, gamma, omega0):
        return ''
    return rf'$\chi={chi:g},\ \gamma={gamma:g},\ \omega_0={omega0:g}$'


def main():
    data = np.load(infile)
    sigmas = data['sigmas']
    Ls = data['Ls']
    z = data[field]

    plot_title = title if title is not None else default_title(data)
    label = cbar_label if cbar_label is not None else FIELD_LABELS.get(field, field)
    default_min, default_max = FIELD_RANGES.get(field, (None, None))
    lo = vmin if vmin is not None else default_min
    hi = vmax if vmax is not None else default_max

    fig, ax = plt.subplots(figsize=figsize)
    # im = ax.pcolormesh(sigmas, Ls, z, cmap=cmap, vmin=vmin, vmax=vmax,
    #                    shading='auto')

    im = plt.pcolormesh(sigmas, Ls, z, cmap=cmap,
                    vmin=lo, vmax=hi, shading='auto',
                    edgecolors='face', linewidth=0, antialiased=True,
                    rasterized=True)   # opcional, mas recomendado dado o tamanho da malha (200x200)
    ax.set_xscale('log')
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_title:
        ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


if __name__ == '__main__':
    main()
