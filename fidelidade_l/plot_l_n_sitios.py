"""
Plota F1(L) para N sítios a partir dos dados gerados por
calcular_l_n_sitios.py em data/n_sitios. Mesmo estilo de
fidelidade_sigma/plot_sigma_n_sitios.py: uma entrada por curva em `curves`
ao chamar main().
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


N = 2
FIELD_LABELS = {
    'F1_pi': r'$F_1(\phi=\pi)$',
    'F1_opt': r'$F_1(\phi_{opt})$',
}


def default_title(data):
    if not all(k in data for k in ('N', 'chi', 'gamma', 'omega0')):
        return ''
    return (rf'$N={int(data["N"])},\ \chi={float(data["chi"]):g},\ '
            rf'\gamma={float(data["gamma"]):g},\ \omega_0={float(data["omega0"]):g}$')


def curve(
    infile,
    field='F1_pi',         # campo do .npz a plotar: 'F1_pi' ou 'F1_opt'
    color='#8A2BE2',       # cor da curva
    label=None,            # rótulo na legenda; se None, usa o padrão de `field`
    marker='o-',           # estilo de marcador+linha (formato do plt.plot)
    linewidth=.5,          # espessura da linha
    markersize=1,          # tamanho do marcador
    markeredgewidth=1.8,   # espessura da borda do marcador
    markeredgecolor=None,  # cor da borda; se None, usa `color` com alpha
    markerfacecolor=None,  # cor de preenchimento; se None, usa `color` com alpha
    marker_alpha=0.5,      # alpha usado quando markeredge/facecolor é None
):
    """Monta os parâmetros de uma curva. Chame uma vez por entrada de `curves`
    para dar a cada gráfico seu próprio estilo."""
    return dict(
        infile=infile, field=field, color=color, label=label, marker=marker,
        linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth,
        markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor,
        marker_alpha=marker_alpha,
    )


def plot_curve(ax, cfg):
    """Plota uma curva (dict criado por `curve(...)`) no eixo `ax`. Retorna os dados carregados."""
    data = np.load(cfg['infile'])
    Ls = data['Ls']
    y = data[cfg['field']]

    label = cfg['label'] if cfg['label'] is not None else FIELD_LABELS.get(cfg['field'], cfg['field'])
    edge_color = cfg['markeredgecolor'] if cfg['markeredgecolor'] is not None else to_rgba(cfg['color'], cfg['marker_alpha'])
    face_color = cfg['markerfacecolor'] if cfg['markerfacecolor'] is not None else to_rgba(cfg['color'], cfg['marker_alpha'])

    ax.plot(
        Ls, y, cfg['marker'],
        color=cfg['color'],
        linewidth=cfg['linewidth'],
        markersize=cfg['markersize'],
        markeredgewidth=cfg['markeredgewidth'],
        markeredgecolor=edge_color,
        markerfacecolor=face_color,
        label=label,
    )
    return data


# ---- Configuração global, uma única vez, no topo do script ----
plt.rcParams.update({
    'figure.figsize': (6, 4.5),
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'svg.fonttype': 'path',
})


def main(
    curves,
    outfile=f'fidelity_L_N{N}.svg',
    dpi=300,
    show=True,
    figsize=(7, 5),
    ylim=(0.3, 1.02),
    xlim=None,
    grid=False,
    title=None,
    xlabel=r'Separação entre os sítios $L\gamma$',
    ylabel='Fidelidade $F_1$',
    legend_loc='upper right',
):
    fig, ax = plt.subplots(figsize=figsize)

    last_data = None
    for c in curves:
        last_data = plot_curve(ax, c)

    plot_title = title if title is not None else (default_title(last_data) if last_data is not None else '')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_title:
        ax.set_title(plot_title)
    ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(loc=legend_loc)
    if grid:
        ax.grid(True, which='both', ls=':')
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def arquivo(sigma, N=N, chi=1000, gamma=1, omega0=0):
    """Caminho do .npz gerado por calcular_l_n_sitios.py."""
    return os.path.join('data', 'n_sitios', f'fidelity_N{N}_sigma{sigma:g}_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}.npz')


if __name__ == '__main__':
    main(
        curves=[
            curve(infile=arquivo(0.184), color='#8A2BE2', label=r'$\sigma/\gamma=0,863$'),
            curve(infile=arquivo(0.158),  color='#FF0000', label=r'$\sigma/\gamma=0,784$'),
            curve(infile=arquivo(1),    color='#3CB371', label=r'$\sigma/\gamma=1$'),
            curve(infile=arquivo(10),   color='#1E90FF', label=r'$\sigma/\gamma=10$'),
        ],
        outfile=f'fidelity_L_N{N}.svg',
    )
