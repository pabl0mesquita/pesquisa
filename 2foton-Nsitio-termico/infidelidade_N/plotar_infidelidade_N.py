"""Plota a infidelidade x N com osciladores lentos a partir de ./data/infidelidade_N.npz
(gerado por calcular_infidelidade_N.py). Nao faz calculo pesado: ajuste o estilo na
secao ESTILO abaixo.

Modelo linear em kappa (validado em I5):  1 - F_max ~ A N^-p + c(N) kappa,  c(N) -> (8/5) N.

I1  1 - F_max x N para kappa fixo            (com kappa fixo existe um N otimo)
I2  1 - F_max x N com kappa = kappa0 N^-alpha (resposta direta: que alpha e preciso)
I3  expoente efetivo p_eff x alpha           (previsao: min(p, alpha - 1))
I4  tolerancia kappa*(N) (e g*/gamma)        (previsao: kappa* ~ A N^-(p+1) / (8/5))
I5  c(N)/N x N                               (convergencia para 8/5)
I6  mapa log10(1 - F_max) no plano (N, kappa)

Uso:
    python plotar_infidelidade_N.py              # todas as figuras
    python plotar_infidelidade_N.py --figs 2 4   # so algumas
    python plotar_infidelidade_N.py --show
"""
import os
import sys
import glob
import argparse
import numpy as np

# =====================================================================
#  ESTILO (edite aqui)
# =====================================================================
DADOS = os.path.join('data', 'infidelidade_N.npz')   # relativo ao diretorio corrente
FORMATOS = [                   # ex.: 'png', 'svg', 'pdf'
    # 'svg',
    'png',
]
DPI = 300
MOSTRAR_TITULO = True
MOSTRAR_MODELO = True          # curvas pontilhadas do modelo A N^-p + c(N) kappa

# Estilo LaTeX sem text.usetex (o PNG exigiria o dvipng, que nao esta instalado):
# matematica em Computer Modern via mathtext e texto em Latin Modern Roman.
RC = {
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'svg.fonttype': 'path',
    'lines.linewidth': 1.2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.frameon': False,
}
FONTES_EXTRA = '/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf'

# Paleta (Mathematica ColorData[97], como em infidelidade/)
CORES = ['#5E81B5', '#E19C24', '#8FB032', '#EB6235', '#8778B3', '#C56E1A', '#5D9EC7']
COR_IDEAL = 'black'
MS = 3.5
LEGENDA_FS = 8
XLABEL_N = r'Número de sítios de interação ($N$)'
YLABEL_INF = r'Infidelidade ($1-\bar F_{\max}$)'

FIGSIZE = {1: (5.4, 4.0), 2: (5.4, 4.0), 3: (5.0, 3.8), 4: (5.4, 4.0), 5: (5.0, 3.6), 6: (5.8, 4.4)}
ARQUIVO = {1: 'figI1_inf_N_kappa_fixo', 2: 'figI2_inf_N_kappa_alpha', 3: 'figI3_p_eff_alpha',
           4: 'figI4_kappa_tolerancia', 5: 'figI5_coef_c', 6: 'figI6_mapa_N_kappa'}

KAPS_FIXOS = [1e-5, 1e-4, 1e-3, 1e-2]   # curvas da fig. I1 (precisam estar na grade de kappa)
FAIXA_AJUSTE = (5, 30)                  # N do ajuste ideal A N^-p
FAIXA_PEFF = (10, 30)                   # N do ajuste do expoente efetivo (fig. I3)
RAZAO_TOL = 1.0                         # kappa*: o oscilador acrescenta RAZAO_TOL x a infidelidade ideal
C_INF = 8 / 5                           # limite de c(N)/N
NBARS_G = [0, 1]                        # eixos g*/gamma da fig. I4 (kappa = (4g)^2 (2 nbar + 1))
CMAP = 'viridis'
NIVEIS_MAPA = [-3, -2.5, -2, -1.5, -1]  # contornos de log10(1-F) na fig. I6


# =====================================================================
#  Infraestrutura
# =====================================================================
def _plt(show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in glob.glob(FONTES_EXTRA):      # Latin Modern do TeX nao esta no caminho do matplotlib
        font_manager.fontManager.addfont(f)
    plt.rcParams.update(RC)
    return plt


def _salvar(fig, k):
    for ext in FORMATOS:
        caminho = os.path.join(os.getcwd(), f'{ARQUIVO[k]}.{ext}')
        fig.savefig(caminho, dpi=DPI, bbox_inches='tight')
        print(f"salvo: {caminho}")


def _titulo(ax, texto):
    if MOSTRAR_TITULO:
        ax.set_title(texto, fontsize=9)


def ajuste(N, y, faixa):
    """y = C N^-p em log-log na faixa de N. Devolve (C, p)."""
    m = (N >= faixa[0]) & (N <= faixa[1])
    a, b = np.polyfit(np.log(N[m]), np.log(y[m]), 1)
    return np.exp(b), -a


def _rot_k(k):
    e = int(np.round(np.log10(k)))
    return fr'$\kappa=10^{{{e}}}$' if np.isclose(k, 10.0 ** e) else fr'$\kappa={k:g}$'


def kappa_estrela(d):
    """kappa*(N): onde (1-F_max) - (1-F0) = RAZAO_TOL (1-F0), interpolando em log-log na grade."""
    K = d['KAPS']; m = K > 0
    out = []
    for i in range(len(d['NS'])):
        I0 = 1 - d['F0'][i]
        dI = (1 - d['F_grade'][i, m]) - I0
        alvo = RAZAO_TOL * I0
        ok = dI > 0
        x, y = np.log(K[m][ok]), np.log(dI[ok])
        j = np.searchsorted(y, np.log(alvo))
        if j == 0 or j >= len(y):
            out.append(np.nan)
        else:
            out.append(np.exp(np.interp(np.log(alvo), y[j - 1:j + 1], x[j - 1:j + 1])))
    return np.array(out)


# =====================================================================
#  Figuras
# =====================================================================
def fig1(plt, d):
    NS, K = d['NS'], d['KAPS']
    A, p = ajuste(NS, 1 - d['F0'], FAIXA_AJUSTE)
    fig, ax = plt.subplots(figsize=FIGSIZE[1])
    ax.loglog(NS, 1 - d['F0'], 'o', color=COR_IDEAL, ms=MS, label=r'$\kappa=0$')
    ax.loglog(NS, A * NS ** -p, '--', color=COR_IDEAL, lw=0.8, label=fr'${A:.3f}\,N^{{-{p:.2f}}}$')
    for c, kf in zip(CORES, KAPS_FIXOS):
        j = int(np.argmin(abs(np.log(K + 1e-300) - np.log(kf))))
        y = 1 - d['F_grade'][:, j]
        ax.loglog(NS, y, 'o-', color=c, ms=MS, mfc='white', lw=1.0, label=_rot_k(K[j]))
        i = int(np.argmin(y))
        ax.loglog(NS[i], y[i], '*', color=c, ms=10, mec='k', mew=0.4)
        if MOSTRAR_MODELO:
            ax.loglog(NS, 1 - d['F0'] + d['c'] * K[j], ':', color=c, lw=0.9)
    ax.set_xlabel(XLABEL_N); ax.set_ylabel(YLABEL_INF)
    ax.legend(fontsize=LEGENDA_FS, loc='lower left', ncol=2)
    _titulo(ax, r'$\kappa$ fixo: estrela $=$ $N$ ótimo; pontilhado: $1-\bar F_0+c(N)\kappa$')
    return fig


def fig2(plt, d):
    NS, al, k0 = d['NS'], d['ALPHAS'], float(d['KAPPA0'])
    A, p = ajuste(NS, 1 - d['F0'], FAIXA_AJUSTE)
    fig, ax = plt.subplots(figsize=FIGSIZE[2])
    ax.loglog(NS, 1 - d['F0'], 'o', color=COR_IDEAL, ms=MS, label=r'$\kappa=0$')
    for c, a, F in zip(CORES, al, d['F_alpha']):
        ax.loglog(NS, 1 - F, 'o-', color=c, ms=MS, mfc='white', lw=1.0, label=fr'$\alpha={a:g}$')
        if MOSTRAR_MODELO:
            ax.loglog(NS, 1 - d['F0'] + d['c'] * k0 * NS ** -a, ':', color=c, lw=0.9)
    ax.set_xlabel(XLABEL_N); ax.set_ylabel(YLABEL_INF)
    ax.legend(fontsize=LEGENDA_FS, loc='lower left', ncol=2)
    _titulo(ax, fr'$\kappa=\kappa_0 N^{{-\alpha}}$, $\kappa_0={k0:g}$; ideal: $p={p:.2f}$')
    return fig


def fig3(plt, d):
    NS, al, k0 = d['NS'], d['ALPHAS'], float(d['KAPPA0'])
    A, p = ajuste(NS, 1 - d['F0'], FAIXA_AJUSTE)
    pe = [ajuste(NS, 1 - F, FAIXA_PEFF)[1] for F in d['F_alpha']]
    af = np.linspace(al.min(), al.max(), 200)
    pm = [ajuste(NS, 1 - d['F0'] + d['c'] * k0 * NS ** -a, FAIXA_PEFF)[1] for a in af]
    fig, ax = plt.subplots(figsize=FIGSIZE[3])
    ax.plot(af, np.minimum(p, af - 1), ':', color='gray', lw=1.0, label=r'$\min(p,\ \alpha-1)$')
    ax.plot(af, pm, '-', color=CORES[0], lw=1.0, label=r'modelo $A N^{-p}+c(N)\kappa$')
    ax.plot(al, pe, 'o', color=CORES[1], ms=5, label='numérico')
    ax.axhline(p, color=COR_IDEAL, ls='--', lw=0.7)
    ax.axhline(0, color='#BBBBBB', lw=0.6)
    ax.set_xlabel(r'$\alpha$  ($\kappa=\kappa_0 N^{-\alpha}$)')
    ax.set_ylabel(r'expoente efetivo $p_{\rm ef}$  ($1-\bar F\propto N^{-p_{\rm ef}}$)')
    ax.legend(fontsize=LEGENDA_FS, loc='lower right')
    _titulo(ax, fr'ajuste em $N\in[{FAIXA_PEFF[0]},{FAIXA_PEFF[1]}]$; tracejado: ideal $p={p:.2f}$')
    return fig


def fig4(plt, d):
    NS = d['NS']
    A, p = ajuste(NS, 1 - d['F0'], FAIXA_AJUSTE)
    ks = kappa_estrela(d)
    klin = RAZAO_TOL * (1 - d['F0']) / d['c']
    kas = RAZAO_TOL * A * NS ** -p / (C_INF * NS)
    m = np.isfinite(ks)
    B, q = ajuste(NS[m], ks[m], FAIXA_AJUSTE)
    fig, ax = plt.subplots(figsize=FIGSIZE[4])
    ax.loglog(NS[m], ks[m], 'o', color=CORES[1], ms=MS + 0.5, label=r'numérico')
    ax.loglog(NS, klin, '-', color=CORES[0], lw=1.0, label=r'linear: $(1-\bar F_0)/c(N)$')
    ax.loglog(NS, kas, '--', color='gray', lw=0.9,
              label=fr'$A N^{{-p}}/(\frac{{8}}{{5}}N)\propto N^{{-{p + 1:.2f}}}$')
    ax.set_xlabel(XLABEL_N); ax.set_ylabel(r'tolerância $\kappa^*$')
    for j, nb in enumerate(NBARS_G):
        fator = 2 * nb + 1
        sec = ax.secondary_yaxis(1.0 + 0.16 * j, functions=(lambda k, f=fator: np.sqrt(np.clip(k, 1e-30, None) / f) / 4,
                                                           lambda g, f=fator: f * (4 * g) ** 2))
        sec.set_ylabel(fr'$g^*/\gamma$ ($\bar n={nb}$)', fontsize=8)
        sec.tick_params(labelsize=7)
    ax.legend(fontsize=LEGENDA_FS, loc='lower left')
    _titulo(ax, fr'$\kappa^*$: oscilador $=$ {RAZAO_TOL:g}$\times$ a infidelidade ideal; '
                fr'ajuste $\kappa^*\approx{B:.2f}\,N^{{-{q:.2f}}}$')
    return fig


def fig5(plt, d):
    NS = d['NS']
    fig, ax = plt.subplots(figsize=FIGSIZE[5])
    ax.semilogx(NS, d['c'] / NS, 'o-', color=CORES[0], ms=MS, lw=1.0)
    ax.axhline(C_INF, color='gray', ls='--', lw=0.9)
    ax.text(NS[0], C_INF + 0.04, r'$8/5$: porta ideal $+$ fases aleatórias', fontsize=8, color='#555555')
    ax.set_xlabel(XLABEL_N)
    ax.set_ylabel(r'$c(N)/N=-\frac{1}{N}\,\partial\bar F_{\max}/\partial\kappa|_0$')
    _titulo(ax, r'coeficiente no $\sigma$ ótimo de $\kappa=0$')
    return fig


def fig6(plt, d):
    from matplotlib.colors import LogNorm  # noqa: F401
    NS, K = d['NS'], d['KAPS']
    m = K > 0
    Z = np.log10(1 - d['F_grade'][:, m]).T          # (kappa, N)
    fig, ax = plt.subplots(figsize=FIGSIZE[6])
    pc = ax.pcolormesh(NS, K[m], Z, shading='gouraud', cmap=CMAP)
    cs = ax.contour(NS, K[m], Z, levels=NIVEIS_MAPA, colors='w', linewidths=0.7)
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=7)
    ks = kappa_estrela(d)
    ax.plot(NS, ks, 'o-', color='#FF5555', ms=2.5, lw=1.0, label=r'$\kappa^*(N)$')
    k0 = float(d['KAPPA0'])
    for a in d['ALPHAS']:
        y = k0 * NS.astype(float) ** -a
        ok = y >= K[m].min()
        ax.plot(NS[ok], y[ok], '--', color='w', lw=0.6)
        if ok.any():
            i = np.where(ok)[0][-1]
            ax.text(NS[i], y[i], fr'$\alpha={a:g}$', color='w', fontsize=7, ha='right', va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(XLABEL_N); ax.set_ylabel(r'$\kappa$')
    plt.colorbar(pc, label=r'$\log_{10}(1-\bar F_{\max})$')
    ax.legend(fontsize=LEGENDA_FS, loc='lower left', labelcolor='w')
    _titulo(ax, fr'tracejado: $\kappa_0 N^{{-\alpha}}$, $\kappa_0={k0:g}$')
    return fig


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Plota infidelidade x N (osciladores lentos).")
    ap.add_argument('--figs', type=int, nargs='+', default=list(FIGS), choices=list(FIGS))
    ap.add_argument('--show', action='store_true', help="mostra as figuras na tela")
    a = ap.parse_args()
    if not os.path.exists(DADOS):
        sys.exit(f"{DADOS} nao existe: rode antes  python calcular_infidelidade_N.py")
    d = dict(np.load(DADOS))
    plt = _plt(a.show)
    A, p = ajuste(d['NS'], 1 - d['F0'], FAIXA_AJUSTE)
    print(f"ideal: 1-F = {A:.3f} N^-{p:.3f}  (BC, chi->inf: 0.537 N^-1.61)")
    for k in a.figs:
        fig = FIGS[k](plt, d)
        fig.tight_layout()
        _salvar(fig, k)
        if not a.show:
            plt.close(fig)
    if a.show:
        plt.show()
