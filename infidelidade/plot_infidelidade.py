"""
Plota a Fig. 3 de Brod & Combes (arXiv:1604.04278) a partir dos dados gerados
por calcular_infidelidade.py em data/, no estilo da figura do artigo (Mathematica).

    Fig. 3(a): F_avg(φ=π) vs σ para vários N.
    Fig. 3(b): 1 - F_max e σ_max vs N (log-log), com ajustes de lei de potência
               y = C N^{-p} feitos aqui, em N ∈ fit_range.

Uso:  python3 plot_infidelidade.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

here = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# Parâmetros do gráfico
# ======================================================================
L = 0.1                 # qual .npz plotar (ver arquivo())
fit_range = (4, 20)     # N usados nos ajustes de lei de potência
N_fig3b_min = 2         # a Fig. 3(b) começa em N = 2

# Paleta padrão do Mathematica (ColorData[97]), na mesma ordem do artigo
MMA = ['#5E81B5', '#E19C24', '#8FB032', '#EB6235', '#8778B3',
       '#C56E1A', '#5D9EC7', '#FFBF00', '#A5609D', '#929600']
xlabel_a = r'Largura do pacote de onda ($\sigma$)'
ylabel_a = r'Fidelidade ($F$)'
xlabel_b = r'Número de sítios de interação ($N$)'
figsize = (5.0, 5.6)
# Altura dos rótulos "N = k" na Fig. 3(a); o x é calculado para o rótulo ficar
# logo à esquerda da subida da curva (como no artigo). Ajuste à mão se precisar.
label_y = {1: 0.56, 2: 0.66, 3: 0.715, 4: 0.77, 5: 0.825, 10: 0.885, 20: 0.925}
label_shift = 1.3      # rótulo centrado em σ_cruzamento / label_shift

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
_tick_font = dict(family='sans-serif', size=8)   # rótulos numéricos sem serifa, como no Mathematica


def arquivo(L, chi=1000, gamma=1, omega0=0):
    """Caminho do .npz gerado por calcular_infidelidade.py."""
    return os.path.join(here, 'data', f'fig3_L{L:g}_chi{chi:g}_gamma{gamma:g}_omega{omega0:g}.npz')


def fit_power_law(N, y, rng=fit_range):
    """Ajuste linear em log-log: y = C N^{-p}. Devolve (C, p)."""
    m = (N >= rng[0]) & (N <= rng[1])
    slope, icpt = np.polyfit(np.log(N[m]), np.log(y[m]), 1)
    return np.exp(icpt), -slope


def _fmt_ticks(ax, xt, xl, yt, yl):
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_xticklabels(xl, **_tick_font)
    ax.set_yticklabels(yl, **_tick_font)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_fig3(r, outfile):
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=figsize,
                                     gridspec_kw=dict(height_ratios=[1.9, 1], hspace=0.28))

    # ---- (a) F vs σ -------------------------------------------------
    sig, Fc = r['sigma_fig3a'], r['F_curves']
    for k, (N, F) in enumerate(zip(r['N_fig3a'], Fc)):
        c = MMA[k % len(MMA)]
        ax_a.semilogx(sig, F, '-', color=c, lw=1.1)
        # rótulo à esquerda da subida da curva, como no artigo
        i_pk = int(np.argmax(F))
        y_lab = min(label_y.get(int(N), 0.7), F[i_pk] - 0.03)
        x_cross = np.exp(np.interp(y_lab, F[:i_pk + 1], np.log(sig[:i_pk + 1])))
        x_lab = max(x_cross / label_shift, sig[0] * 1.3)
        ax_a.text(x_lab, y_lab, f'N = {int(N)}', ha='center', va='center',
                  fontdict=dict(_tick_font, size=7),
                  bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.6',
                            fc='#E6E6E6', ec='none'))
    ax_a.set_xlim(sig[0], sig[-1])
    ax_a.set_ylim(0.35, 1.0)
    _fmt_ticks(ax_a, [0.01, 0.05, 0.1, 0.5, 1], ['0.01', '0.05', '0.1', '0.5', '1'],
               np.arange(0.4, 1.01, 0.1), [f'{v:.1f}' for v in np.arange(0.4, 1.01, 0.1)])
    ax_a.set_xlabel(xlabel_a, labelpad=2)
    ax_a.set_ylabel(ylabel_a)
    ax_a.text(0.975, 0.965, '(a)', transform=ax_a.transAxes, ha='right', va='top',
              fontdict=dict(_tick_font, size=9))

    # ---- (b) 1-F_max e σ_max vs N -------------------------------------
    m = r['N'] >= N_fig3b_min
    N, smax, inf = r['N'][m], r['sigma_max'][m], 1 - r['F_max'][m]
    B, q = fit_power_law(r['N'], r['sigma_max'])
    A, p = fit_power_law(r['N'], 1 - r['F_max'])
    Nf = np.linspace(N.min(), N.max() + 0.6, 100)
    ax_b.loglog(Nf, B * Nf ** -q, '--', color=MMA[0], lw=1.0, dashes=(4, 2.5))
    ax_b.loglog(Nf, A * Nf ** -p, '--', color=MMA[1], lw=1.0, dashes=(4, 2.5))
    ax_b.loglog(N, smax, 'o', color=MMA[0], ms=4.2, mec='none', label=r'$\sigma_{\max}$')
    ax_b.loglog(N, inf, 'o', mfc='white', mec=MMA[1], mew=0.9, ms=4.2, label=r'$1-F$')
    ax_b.set_xlim(N.min() * 0.93, N.max() * 1.06)
    lo = min(smax.min(), inf.min())
    hi = max(smax.max(), inf.max())
    yt = [v for v in (0.001, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.200, 0.500)
          if lo / 1.3 <= v <= hi * 1.3]
    ax_b.set_ylim(lo / 1.3, hi * 1.3)
    _fmt_ticks(ax_b, [2, 5, 10, 20], ['2', '5', '10', '20'], yt, [f'{v:.3f}' for v in yt])
    ax_b.set_xlabel(xlabel_b, labelpad=2)
    ax_b.legend(loc='lower left', handletextpad=0.1, borderaxespad=0.3,
                labelspacing=0.3, fontsize=10)
    ax_b.text(0.975, 0.93, '(b)', transform=ax_b.transAxes, ha='right', va='top',
              fontdict=dict(_tick_font, size=9))

    print(f'Ajustes em N ∈ {list(fit_range)} (L={float(r["L"]):g}):  1-F = {A:.3f} N^(-{p:.3f}),'
          f'  σ_max = {B:.3f} N^(-{q:.3f})')
    print('Artigo (L=0, χ→∞):            1-F = 0.537 N^(-1.61),  σ_max = 0.350 N^(-0.81)')
    fig.savefig(outfile, bbox_inches='tight')
    print(f'Figura: {outfile}')
    return fig


def main(infile=arquivo(L), outfile=None, show=True):
    r = dict(np.load(infile))
    if outfile is None:
        outfile = os.path.join(here, f'fig3_L{float(r["L"]):g}_Nmax{int(r["N"].max())}.svg')
    fig = plot_fig3(r, outfile)
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    main(infile=arquivo(L))
