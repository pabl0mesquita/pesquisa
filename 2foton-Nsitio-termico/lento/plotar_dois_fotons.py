"""Plota as figuras D1-D5 a partir de ./data/dois_fotons_dados.npz (gerado por
calcular_dois_fotons.py). Nao faz nenhum calculo: ajuste livremente o estilo na
secao ESTILO abaixo.

D1  Reproducao de BC Fig. 3(a) (kappa=0), N=1,3,5
D2  F x sigma estilo Fig. 3(a), 4 paineis kappa = 0, 1e-3, 1e-2, 5e-2
D3  F x sigma, um painel por N, varias kappa sobrepostas
D4  F_max em funcao de kappa
D5  coeficiente c(sigma) = -(1/N) dF/dkappa|_0

Uso:
    python plotar_dois_fotons.py              # todas as figuras
    python plotar_dois_fotons.py --figs 1 4   # so algumas
    python plotar_dois_fotons.py --show       # abre as janelas alem de salvar
"""
import os
import sys
import glob
import argparse
import numpy as np

# =====================================================================
#  ESTILO (edite aqui)
# =====================================================================
DADOS = os.path.join('data', 'dois_fotons_dados.npz')   # relativo ao diretorio corrente
FORMATOS = [                   # ex.: 'png', 'svg', 'pdf'
    # 'svg',
    'png',
]
DPI = 300
MOSTRAR_TITULO = True

# Estilo LaTeX sem text.usetex (o PNG exigiria o dvipng, que nao esta instalado):
# matematica em Computer Modern via mathtext e texto em Latin Modern Roman
# (registrada a partir do TeX em _plt; se faltar, cai em DejaVu Serif).
RC = {
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'svg.fonttype': 'path',
    'lines.linewidth': 1.2,
}
FONTES_EXTRA = '/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf'

ESTILO_N = {1: ('#A51C1C', 'o'), 3: ('#2FA82F', 'D'), 5: ('#2AB3BE', '^')}   # (cor, marcador) por N
KCOR = {0: 'k', 1e-3: '#1f77b4', 1e-2: '#2ca02c', 5e-2: '#d62728'}          # cor por kappa (fig. D3)
LW = 1.6                       # espessura das curvas principais
COR_MOLDURA = '#888888'
TICK_FS = 9

FIGSIZE = {1: (5.6, 3.9), 2: (8.4, 6.2), 3: (10.5, 3.4), 4: (6.0, 4.0), 5: (5.6, 3.7)}
ARQUIVO = {1: 'figD1_BC_kappa0', 2: 'figD2_F_sigma_kappas', 3: 'figD3_por_N',
           4: 'figD4_Fmax_kappa', 5: 'figD5_coef'}
YLIM_F = (0.35, 1.0)           # eixo y de F (figs. D1-D3)
ANOTACAO_D1 = {1: (-88, 0), 3: (8, 4), 5: (-72, 2)}   # deslocamento (pt) do rotulo F_max por N


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


def _moldura(a):
    a.tick_params(direction='in', which='both', top=True, right=True, labelsize=TICK_FS)
    for sp in a.spines.values():
        sp.set_color(COR_MOLDURA)


def _curva(a, x, y, N, **extra):
    kw = dict(color=ESTILO_N[N][0], lw=LW)
    kw.update(extra)
    a.plot(x, y, **kw)


def _legN(fig, NS):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ESTILO_N[N][0], lw=2.2) for N in NS]
    fig.legend(h, [str(N) for N in NS], title=r'$N$', loc='center right', frameon=False,
               fontsize=10, title_fontsize=12, handletextpad=0.3)


# =====================================================================
#  Figuras
# =====================================================================
def fig1(plt, d):
    NS, SIGS = [int(n) for n in d['NS']], d['SIGS']
    fig, a = plt.subplots(figsize=FIGSIZE[1])
    for i, N in enumerate(NS):
        y = d['R'][0, i]
        _curva(a, SIGS, y, N)
        j = np.argmax(y)
        a.annotate(fr'$F_{{\max}}={y[j]:.3f}$', (SIGS[j], y[j]), textcoords='offset points',
                   xytext=ANOTACAO_D1.get(N, (8, 4)), fontsize=8, color=ESTILO_N[N][0])
    a.axhline(0.4, color='#999999', ls='--', lw=0.8)
    a.text(0.0035, 0.375, r'$F=0.4$ (identidade $\times$ CZ)', fontsize=8, color='#666666')
    a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$'); a.set_ylabel(r'$\bar F$')
    a.set_ylim(YLIM_F[0], 1.02); _moldura(a)
    if MOSTRAR_TITULO:
        a.set_title(r'Sem oscilador ($\kappa=0$): reprodução de BC, Fig. 3(a)', fontsize=10)
    _legN(fig, NS)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return fig


def fig2(plt, d):
    NS, SIGS, KAPS = [int(n) for n in d['NS']], d['SIGS'], d['KAPS']
    fig, ax = plt.subplots(2, 2, figsize=FIGSIZE[2], sharex=True, sharey=True)
    for ik, k in enumerate(KAPS):
        a = ax.flat[ik]
        for i, N in enumerate(NS):
            if k > 0:
                a.plot(SIGS, d['R'][0, i], color=ESTILO_N[N][0], lw=0.8, ls='--', alpha=0.6)
            _curva(a, SIGS, d['R'][ik, i], N)
            a.text(0.04, 0.82 - 0.075 * i, fr'$N={N}$: $F_{{\max}}={d["R"][ik, i].max():.3f}$',
                   transform=a.transAxes, fontsize=8.5, color=ESTILO_N[N][0])
        a.set_xscale('log'); a.set_ylim(*YLIM_F); _moldura(a)
        a.text(0.04, 0.92, (r'$\kappa=0$ (BC)' if k == 0 else fr'$\kappa={k:g}$'),
               transform=a.transAxes, fontsize=10)
        a.axhline(0.4, color='#BBBBBB', lw=0.7)
    for a in ax[1]:
        a.set_xlabel(r'$\sigma/\gamma$')
    for a in ax[:, 0]:
        a.set_ylabel(r'$\bar F$')
    _legN(fig, NS)
    if MOSTRAR_TITULO:
        fig.suptitle(r'CZ contrapropagante com osciladores lentos (tracejado: $\kappa=0$)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    return fig


def fig3(plt, d):
    NS, SIGS, KAPS = [int(n) for n in d['NS']], d['SIGS'], d['KAPS']
    fig, ax = plt.subplots(1, len(NS), figsize=FIGSIZE[3], sharey=True)
    for i, N in enumerate(NS):
        a = ax[i]
        for ik, k in enumerate(KAPS):
            cor = KCOR.get(float(k), None)
            a.plot(SIGS, d['R'][ik, i], color=cor, lw=LW, label=(r'$0$' if k == 0 else f'{k:g}'))
            a.text(0.04, 0.92 - 0.075 * ik,
                   (r'$\kappa=0$' if k == 0 else fr'$\kappa={k:g}$') + fr': $F_{{\max}}={d["R"][ik, i].max():.3f}$',
                   transform=a.transAxes, fontsize=8, color=cor)
        a.set_xscale('log'); a.set_title(fr'$N={N}$', fontsize=10); a.set_xlabel(r'$\sigma/\gamma$')
        a.set_ylim(*YLIM_F); _moldura(a)
    ax[0].set_ylabel(r'$\bar F$')
    ax[-1].legend(title=r'$\kappa$', fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout()
    return fig


def fig4(plt, d):
    NS, SIGS = [int(n) for n in d['NS']], d['SIGS']
    kg, R2, sf, D = d['kgrid'], d['R2'], d['sfin'], d['D']
    fig, a = plt.subplots(figsize=FIGSIZE[4])
    for i, N in enumerate(NS):
        Fm = R2[:, i].max(axis=1)
        s0 = sf[i][np.argmax(R2[0, i])]
        c = -np.interp(np.log(s0), np.log(SIGS), D[i])          # -dF/dkappa no sigma otimo
        x = kg[1:]
        _curva(a, x, 1 - Fm[1:], N, ls='-')
        a.plot(x, (1 - Fm[0]) + c * x, color=ESTILO_N[N][0], ls='--', lw=0.9)
        a.axhline(1 - Fm[0], color=ESTILO_N[N][0], ls=':', lw=0.8)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel(r'$\kappa$'); a.set_ylabel(r'$1-\bar F_{\max}$')
    if MOSTRAR_TITULO:
        a.set_title(r'tracejado: $1-\bar F_{\max}^{(0)}+c\,\kappa$; pontilhado: $\kappa=0$', fontsize=9)
    _moldura(a); _legN(fig, NS)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return fig


def fig5(plt, d):
    NS, SIGS = [int(n) for n in d['NS']], d['SIGS']
    fig, a = plt.subplots(figsize=FIGSIZE[5])
    for i, N in enumerate(NS):
        _curva(a, SIGS, -d['D'][i] / N, N)
    a.axhline(1.6, color='#777777', ls='--', lw=0.9)
    a.text(0.0035, 1.65, r'$+8/5$: porta ideal + fases aleatórias', fontsize=8, color='#555555')
    a.axhline(-0.8, color='#777777', ls=':', lw=0.9)
    a.text(0.0035, -0.95, r'$-4/5$: sem interação ($\lambda=-1$) + fases', fontsize=8, color='#555555')
    a.axhline(0, color='#BBBBBB', lw=0.6)
    a.set_ylim(-1.05, 1.85)
    a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$')
    a.set_ylabel(r'$c(\sigma)/N=-\frac{1}{N}\,\partial\bar F/\partial\kappa|_{0}$')
    _moldura(a); _legN(fig, NS)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return fig


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Plota as figuras D1-D5 a partir de data/.")
    ap.add_argument('--figs', type=int, nargs='+', default=list(FIGS), choices=list(FIGS))
    ap.add_argument('--show', action='store_true', help="mostra as figuras na tela")
    a = ap.parse_args()
    if not os.path.exists(DADOS):
        sys.exit(f"{DADOS} nao existe: rode antes  python calcular_dois_fotons.py")
    d = dict(np.load(DADOS))
    plt = _plt(a.show)
    for k in a.figs:
        fig = FIGS[k](plt, d)
        _salvar(fig, k)
        if not a.show:
            plt.close(fig)
    if a.show:
        plt.show()
