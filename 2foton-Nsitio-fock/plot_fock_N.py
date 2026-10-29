# -*- coding: utf-8 -*-
"""
Plota a Fig. 3(a) de Brod & Combes (arXiv:1604.04278) -- fidelidade F_1 em
funcao da largura sigma do pacote para varios N -- no cenario com um
oscilador em estado de Fock em cada atomo, a partir dos .npz gerados por
calcular_fock_N.py.  Mesmas cores do artigo (paleta padrao do Mathematica,
a mesma de infidelidade/plot_infidelidade.py).

Gera uma figura por caso listado em `casos_plot` (calcular_fock_N.py, soma
truncada em m) e outra por caso em `casos_lenta` (calcular_fock_N_lenta.py,
regime de mola lenta omega << gamma, com a faixa do erro estatistico).

Uso:  python3 plot_fock_N.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

here = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# Parametros do grafico
# ======================================================================
N_values = [1, 2, 3, 4, 5, 10, 20]   # curvas (as mesmas da Fig. 3(a))
pars = dict(chi=1000, gam=1, omega0=0)   # parametros dos sitios usados em calcular_fock_N.py
max_excitados = 2                    # truncamento usado no calculo
referencia = 'elastica'
# casos com osciladores (w_a, S_a, n_a, w_b, S_b, n_b), um grafico por caso
casos_plot = [
    (1.0, 0.01, 0, 1.0, 0.01, 0),
    (1.0, 0.01, 1, 1.0, 0.01, 1),
    (1.0, 0.01, 2, 1.0, 0.01, 2),
]
# casos de mola lenta (g_a, n_a, g_b, n_b), um grafico por caso
casos_lenta = [
    (0.05, 0, 0.05, 0),
    (0.05, 1, 0.05, 1),
    (0.05, 2, 0.05, 2),
]

# Paleta padrao do Mathematica (ColorData[97]), na mesma ordem do artigo
MMA = ['#5E81B5', '#E19C24', '#8FB032', '#EB6235', '#8778B3',
       '#C56E1A', '#5D9EC7', '#FFBF00', '#A5609D', '#929600']
xlabel = r'Largura do pacote de onda ($\sigma$)'
ylabel = r'Fidelidade ($F$)'
figsize = (5.0, 3.9)
xlim = (0.01, 1.0)
ylim = (0.2, 1.0)
label_shift = 1.3      # rotulo "N = k" centrado em sigma_cruzamento / label_shift
label_dy = 0.04        # rotulo fica label_dy abaixo do pico; desce mais se colidir com outro
label_dx_dec = 0.16    # distancia minima (em decadas de sigma) entre rotulos na mesma altura

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'cm', 'font.size': 10, 'svg.fonttype': 'path',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#555555',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.color': '#555555', 'ytick.color': '#555555',
    'xtick.labelcolor': 'black', 'ytick.labelcolor': 'black',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.frameon': False,
})
_tick_font = dict(family='sans-serif', size=8)   # numeros sem serifa, como no Mathematica


def arquivo(N, w_a, S_a, n_a, w_b, S_b, n_b, exc=max_excitados):
    """Caminho do .npz gerado por calcular_fock_N.py (mesmo formato de nome_arquivo)."""
    return os.path.join(here, 'data',
                        f"fid_N{N}_chi{pars['chi']:g}_gam{pars['gam']:g}_w0{pars['omega0']:g}"
                        f"_wa{w_a:g}_Sa{S_a:g}_na{n_a}_wb{w_b:g}_Sb{S_b:g}_nb{n_b}"
                        f"_exc{exc}_{referencia}.npz")


def arquivo_lenta(N, g_a, n_a, g_b, n_b):
    """Caminho do .npz gerado por calcular_fock_N_lenta.py (mesmo formato de nome_arquivo)."""
    return os.path.join(here, 'data',
                        f"fidlenta_N{N}_chi{pars['chi']:g}_gam{pars['gam']:g}_w0{pars['omega0']:g}"
                        f"_ga{g_a:g}_na{n_a}_gb{g_b:g}_nb{n_b}_{referencia}.npz")


def posicao_rotulo(sig, F, y_lab):
    """(x, y) do rotulo "N = k" logo a esquerda da subida da curva, na altura y_lab."""
    i_pk = int(np.argmax(F))
    if i_pk == 0 or y_lab <= F[0]:
        return sig[i_pk], F[i_pk] + 0.03
    x_cross = np.exp(np.interp(y_lab, F[:i_pk + 1], np.log(sig[:i_pk + 1])))
    return max(x_cross / label_shift, sig[0] * 1.3), y_lab


def rotulos(ax, curvas):
    """Caixas cinzas "N = k", como no artigo.  Se um rotulo colide com um ja
    colocado, ele desce ao longo da propria curva ate ficar livre."""
    feitos = []
    for N, sig, F, cor in curvas:
        y = F.max() - label_dy
        for _ in range(12):
            x, y_ = posicao_rotulo(sig, F, y)
            livre = all(abs(np.log10(x / x2)) > label_dx_dec or abs(y_ - y2) > 0.045
                        for x2, y2 in feitos)
            if livre:
                break
            y -= 0.035
        feitos.append((x, y_))
        ax.text(x, y_, f'N = {N}', ha='center', va='center',
                fontdict=dict(_tick_font, size=7), color=cor,
                bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.6', fc='#E6E6E6', ec='none'))


def plot_caso(caso, outfile, show=False, lenta=False):
    """Uma figura para o caso `caso`.  lenta=True: caso (g_a, n_a, g_b, n_b) do
    regime de mola lenta, com a faixa de +-1 erro estatistico."""
    fig, ax = plt.subplots(figsize=figsize)
    curvas = []
    for k, N in enumerate(N_values):
        cor = MMA[k % len(MMA)]
        f = arquivo_lenta(N, *caso) if lenta else arquivo(N, *caso)
        if not os.path.exists(f):
            prog = 'calcular_fock_N_lenta.py' if lenta else 'calcular_fock_N.py'
            print(f'  faltando {os.path.basename(f)} (rode {prog})')
            continue
        d = np.load(f)
        ax.semilogx(d['sigmas'], d['F1'], '-', color=cor, lw=1.1)
        if lenta:
            ax.fill_between(d['sigmas'], d['F1'] - d['erro'], d['F1'] + d['erro'],
                            color=cor, alpha=0.25, lw=0)
        curvas.append((N, d['sigmas'], d['F1'], cor))
    if not curvas:
        print(f'  nenhum dado para o caso {caso}: figura nao gerada')
        plt.close(fig)
        return
    rotulos(ax, curvas)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.05, 0.1, 0.5, 1]))
    ax.set_xticklabels(['0.01', '0.05', '0.1', '0.5', '1'], **_tick_font)
    yt = np.round(np.arange(np.ceil(ylim[0] * 10) / 10, ylim[1] + 1e-9, 0.1), 1)
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_yticklabels([f'{v:.1f}' for v in yt], **_tick_font)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel)
    base = rf"\chi={pars['chi']:g},\ \gamma={pars['gam']:g},\ \omega_0={pars['omega0']:g},\ "
    if lenta:
        g_a, n_a, g_b, n_b = caso
        ax.set_title(rf"${base}\omega^{{(a,b)}}\ll\gamma,\ g={g_a:g},\ n={n_a}$", fontsize=9)
    else:
        w_a, S_a, n_a, w_b, S_b, n_b = caso
        ax.set_title(rf"${base}\omega^{{(a,b)}}={w_a:g},\ S={S_a:g},\ n={n_a}$", fontsize=9)
    ax.text(0.975, 0.965, '(a)', transform=ax.transAxes, ha='right', va='top',
            fontdict=dict(_tick_font, size=9))
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches='tight')
    fig.savefig(outfile.replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    print(f'Figura: {outfile}')
    if show:
        plt.show()
    plt.close(fig)


def main(show=False):
    for caso in casos_plot:
        w_a, S_a, n_a, w_b, S_b, n_b = caso
        plot_caso(caso, os.path.join(here, f'fig3a_fock_wa{w_a:g}_S{S_a:g}_n{n_a}.svg'), show)
    for caso in casos_lenta:
        g_a, n_a, g_b, n_b = caso
        plot_caso(caso, os.path.join(here, f'fig3a_lenta_g{g_a:g}_n{n_a}.svg'), show, lenta=True)


if __name__ == '__main__':
    main()
