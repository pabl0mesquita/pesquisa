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
outfile = 'heatmap_overlap_sigma_L.svg'   # arquivo de imagem de saída
dpi = 300             # resolução da imagem salva
show = True           # se True, abre a janela interativa além de salvar

# ----------------------------------------------------------------------
# O que colorir
# ----------------------------------------------------------------------
field = 'F_abs'       # 'F_abs' (produto interno), 'F1_pi' ou 'F1_opt' (fidelidade)
vmin, vmax = 0.0, 1.0  # faixa da barra de cores
cbar_label = None     # rótulo da barra; se None, usa o padrão de `field`

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
    'F1_pi': r'$F_1(\phi=\pi)$',
    'F1_opt': r'$F_1(\phi_{opt})$',
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

    fig, ax = plt.subplots(figsize=figsize)
    # im = ax.pcolormesh(sigmas, Ls, z, cmap=cmap, vmin=vmin, vmax=vmax,
    #                    shading='auto')

    im = plt.pcolormesh(sigmas, Ls, z, cmap=cmap,
                    vmin=vmin, vmax=vmax, shading='auto',
                    edgecolors='face', linewidth=0, antialiased=False,
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
