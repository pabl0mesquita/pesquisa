"""
Plota F1(sigma) a partir dos dados gerados por calcular.py em
data/counterprop. Suporta várias curvas (de arquivos .npz diferentes,
ou campos diferentes do mesmo arquivo) na mesma imagem: basta listar
uma entrada por curva em `curves` ao chamar main().
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

FIELD_LABELS = {
    'F1_pi': r'$F_1(\phi=\pi)$ — 2 sítios, contrapropagante',
    'F1_opt': r'$F_1(\phi_{opt})$ — 2 sítios, contrapropagante',
}


def default_title(data):
    chi1 = float(data['chi1']) if 'chi1' in data else None
    gamma1 = float(data['gamma1']) if 'gamma1' in data else None
    omega0 = float(data['omega0']) if 'omega0' in data else None
    L = float(data['L']) if 'L' in data else None
    if None in (chi1, gamma1, omega0, L):
        return ''
    return rf'$\chi=1000,\ \gamma=1,\ \omega_0=0$'

    #\gamma={gamma1:.3g}
    #\omega_0={omega0:.3g}
def curve(
    infile,
    field='F1_pi',         # campo do .npz a plotar: 'F1_pi' ou 'F1_opt'
    color='#8A2BE2',       # cor da curva
    label=None,            # rótulo na legenda; se None, usa o padrão de `field`
    marker='D-',           # estilo de marcador+linha (formato do plt.plot)
    linewidth=.5,           # espessura da linha
    markersize=1,          # tamanho do marcador
    markeredgewidth=2.5,   # espessura da borda do marcador
    markeredgecolor=None,  # cor da borda; se None, usa `color` com alpha
    markerfacecolor=None,  # cor de preenchimento; se None, usa `color` com alpha
    marker_alpha=0.4,      # alpha usado quando markeredge/facecolor é None
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
    sigmas = data['sigmas']
    y = data[cfg['field']]

    label = cfg['label'] if cfg['label'] is not None else FIELD_LABELS.get(cfg['field'], cfg['field'])
    edge_color = cfg['markeredgecolor'] if cfg['markeredgecolor'] is not None else to_rgba(cfg['color'], cfg['marker_alpha'])
    face_color = cfg['markerfacecolor'] if cfg['markerfacecolor'] is not None else to_rgba(cfg['color'], cfg['marker_alpha'])

    ax.semilogx(
        sigmas, y, cfg['marker'],
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
    'mathtext.fontset': 'cm',      # escolha UMA opção, sem repetir a chave
    'axes.linewidth': 1.0,
    'svg.fonttype': 'path',   # <-- em vez de 'none'

})

def main(
    curves,
    outfile='fidelity_counterprop.png',
    dpi=300,
    show=True,
    figsize=(7, 5),
    ylim=(0.3, 1.02),
    xlim=(1e-2, 1e2),
    grid=False,
    title=None,
    xlabel=r'Largura do pacote de onda $\sigma$',
    ylabel='Fidelidade $F_1$',
    legend_loc='upper right',
):
    fig, ax = plt.subplots(figsize=figsize)

    last_data = None
    for curve in curves:
        last_data = plot_curve(ax, curve)

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


if __name__ == '__main__':
    main(
        curves=[
            curve(
                infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l0_chi1000_gamma1_omega0_min001.npz'),
                field='F1_pi',
                color='#8A2BE2',
                label=r'$\frac{L}{\gamma}=0$', 
                marker='o-', 
                linewidth=.5, 
                markersize=1, 
                markeredgewidth=2.7, 
                markeredgecolor=None, 
                markerfacecolor=None, 
                marker_alpha=0.5
            ),
            # curve(
            #     infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l01_chi10_gamma1_omega0_min001.npz'),
            #     field='F1_pi',
            #     color='#8A2BE2',
            #     label="L=0.1", 
            #     marker='--', 
            #     linewidth=.5, 
            #     markersize=1, 
            #     markeredgewidth=3, 
            #     markeredgecolor=None, 
            #     markerfacecolor=None, 
            #     marker_alpha=0.4
            # ),
            # curve(
            #     infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l01_chi10_gamma1_omega0.npz'),
            #     field='F1_pi',
            #     color='#8A2BE2',
            #     label="L=0.1", 
            #     marker='o-', 
            #     linewidth=.5, 
            #     markersize=1, 
            #     markeredgewidth=3, 
            #     markeredgecolor=None, 
            #     markerfacecolor=None, 
            #     marker_alpha=0.4
            # ),
            # adicione mais chamadas curve(...) aqui para sobrepor outras curvas/arquivos
            curve(
                infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l04_chi1000_gamma1_omega0_min001.npz'),
                field='F1_pi',
                color='#FF0000',
               label=r'$\frac{L}{\gamma}=0.4$',
                marker='o-', 
                linewidth=.5, 
                markersize=1, 
                markeredgewidth=2.7, 
                markeredgecolor=None, 
                markerfacecolor=None,
                marker_alpha=0.5
            ),
             curve(
                infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l06_chi1000_gamma1_omega0_min001.npz'),
                field='F1_pi',
                color='#3CB371',
                label=r'$\frac{L}{\gamma}=0.6$',
                marker='o-', 
                linewidth=.5, 
                markersize=1, 
                markeredgewidth=2.7 ,
                markeredgecolor=None, 
                markerfacecolor=None,
                marker_alpha=0.5
            ),
            curve(
                infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l08_chi1000_gamma1_omega0_min001.npz'),
                field='F1_pi',
                color='#FF8C00',
                label=r'$\frac{L}{\gamma}=0.8$',
                marker='o-', 
                linewidth=.5, 
                markersize=1, 
                markeredgewidth=2.7, 
                markeredgecolor=None, 
                markerfacecolor=None,
                marker_alpha=0.5
            ),
               curve(
                infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l1_chi1000_gamma1_omega0_min001.npz'),
                field='F1_pi',
                color='#1E90FF',
                label=r'$\frac{L}{\gamma}=1.0$',
                marker='o-', 
                linewidth=.5, 
                markersize=1, 
                markeredgewidth=2.7, 
                markeredgecolor=None, 
                markerfacecolor=None,
                marker_alpha=0.5
            ),
             curve(
                infile=os.path.join('data', 'counterprop', 'teste_fidelity_counterprop_l2_chi1000_gamma1_omega0_min001.npz'),
                field='F1_pi',
                color='black',
                label=r'$\frac{L}{\gamma}=2.0$',
                marker='o-', 
                linewidth=.5, 
                markersize=1, 
                markeredgewidth=2.7, 
                markeredgecolor=None, 
                markerfacecolor=None,
                marker_alpha=0.5
            ),
            
        ],
        outfile='fidelity_all.svg',
    )
