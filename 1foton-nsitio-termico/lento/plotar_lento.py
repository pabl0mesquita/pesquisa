"""Plota as figuras L1-L6 do regime lento a partir de ./data/figL{k}.npz (gerados por
calcular_lento.py). So usa as formulas fechadas de lento.py (baratas); ajuste livremente
o estilo na secao ESTILO abaixo.

Uso:
    python plotar_lento.py              # todas as figuras com dados disponiveis
    python plotar_lento.py --figs 2 6   # so algumas
    python plotar_lento.py --show       # abre as janelas alem de salvar
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lento import C1_fechado  # noqa: E402

# =====================================================================
#  ESTILO (edite aqui)
# =====================================================================
PASTA = 'data'                 # onde estao os .npz (relativa ao diretorio corrente)
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

CN = {1: 'k', 2: 'b', 3: 'c', 5: 'r', 10: 'g', 20: 'm'}   # cor por numero de sitios N
CORES_G = ['k', 'b', 'r', 'g']       # fig. L2 (uma por g)
CORES_NB = ['k', 'r']                # fig. L6 esq. (uma por nbar)
CORES_SIG = ['b', 'g']               # fig. L6 dir. (uma por sigma)
LEGENDA_FS = 7
TITULO_FS = 10
MS = 2.5

FIGSIZE = {1: (10, 7.2), 2: (10, 3.9), 3: (6.4, 4.8), 4: (10, 3.9), 5: (10, 3.9), 6: (10, 3.9)}
ARQUIVO = {1: 'figL1_vs_freq', 2: 'figL2_vs_N', 3: 'figL3_mapa_N10', 4: 'figL4_vs_sigma',
           5: 'figL5_vs_T', 6: 'figL6_relacao_validacao'}

# fig. L3
CMAP = 'viridis'
NIVEIS_F = [0.9, 0.99, 0.999]
NIVEIS_X = [0.015, 0.15]             # contornos vermelhos de N kappa  (2/3 N kappa = 0.01; 0.1)


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


def _dados(k):
    return np.load(os.path.join(PASTA, f'figL{k}.npz'))


def _salvar(fig, k):
    for ext in FORMATOS:
        caminho = os.path.join(os.getcwd(), f'{ARQUIVO[k]}.{ext}')
        fig.savefig(caminho, dpi=DPI, bbox_inches='tight')
        print(f"salvo: {caminho}")


def _titulo(fig, texto, fs=TITULO_FS):
    if MOSTRAR_TITULO:
        fig.suptitle(texto, fontsize=fs)


# =====================================================================
#  Figuras
# =====================================================================
def fig1(plt):
    d = _dados(1)
    nus, Ns = d['nus'], d['Ns']
    wb, g, sig, T, nb, k, S = (float(d[x]) for x in ('wb', 'g', 'sig', 'T', 'nb', 'kappa', 'S'))
    fig, ax = plt.subplots(2, 2, figsize=FIGSIZE[1])
    for i, N in enumerate(Ns):
        c = CN[int(N)]; lab = fr'$N={N}$'
        ax[0, 0].plot(nus, d['eta'][i], c, label=lab)
        ax[0, 1].plot(nus, abs(d['C'][i]), c, label=lab)
        ax[1, 0].plot(nus, d['Fbar'][i], c, label=lab); ax[1, 0].plot(nus, d['Fopt'][i], c+'--', lw=1)
        ax[1, 1].plot(nus, np.angle(d['C'][i])/np.pi, c, label=fr'$\arg C$, $N={N}$')
    ax2 = ax[1, 1].twinx()
    ax2.plot(nus, np.mod(d['phase'][0], 2*np.pi)/np.pi, color='gray', lw=2.5, alpha=0.35)
    ax2.set_ylabel(r'fase ideal de 1 sítio $\arg t_0/\pi$ (cinza)', color='gray')
    ax2.tick_params(axis='y', colors='gray'); ax2.set_ylim(-0.05, 2.05)
    ax[1, 1].set_ylim(-1.05, 1.05)
    ax[1, 0].axhline(2/3, color='gray', lw=0.6, ls=':')
    for a in ax.flat:
        a.axvline(0, color='gray', lw=0.5, ls=':'); a.set_xlabel(r'$\omega_c/\gamma$')
        a.legend(fontsize=LEGENDA_FS)
    ax[0, 0].set_ylabel(r'$\eta$ (população no modo ideal)')
    ax[0, 1].set_ylabel(r'$|C|$ (coerência)')
    ax[1, 0].set_ylabel(r'$\bar F$ (cheia) e com correção de fase (tracejada)')
    ax[1, 1].set_ylabel(r'erro de fase $\arg C/\pi$')
    _titulo(fig, fr'Regime lento; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$ ($S={S:.0f}$), '
                 fr'$k_BT={T:g}\hbar\gamma$ ($\bar n={nb:.2f}$), $\kappa={k:.3f}$, $\sigma={sig:g}\gamma$')
    return fig


def fig2(plt):
    d = _dados(2)
    Ns, ks = d['Ns'], d['kappas']
    wb, sig, T, nb = (float(d[x]) for x in ('wb', 'sig', 'T', 'nb'))
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE[2])
    for i, (k, c) in enumerate(zip(ks, CORES_G)):
        ax[0].plot(Ns, 1-d['Fopt'][i], c, marker='o', ms=2.0, label=fr'$\kappa={k:.4f}$')
        ax[0].plot(Ns, (2/3)*Ns*k, c+'--', lw=1)
        ax[0].plot(Ns, 1-(2+C1_fechado(k)**Ns)/3, c+':', lw=1.4)
        ax[1].plot(Ns, d['eta'][i], c, label=fr'$\eta$, $\kappa={k:.4f}$')
        ax[1].plot(Ns, d['absC'][i], c+'--', lw=1)
    ax[0].axhline(1/3, color='gray', lw=0.6)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    for a in ax:
        a.set_xlabel(r'$N$'); a.legend(fontsize=LEGENDA_FS)
    ax[0].set_ylabel(r'$1-\bar F$')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    _titulo(fig, fr'Regime lento, na linha; $\omega_b={wb:g}\gamma$, $k_BT={T:g}\hbar\gamma$ '
                 fr'($\bar n={nb:.2f}$), $\sigma={sig:g}\gamma$. Esq.: tracejada $=\frac{{2}}{{3}}N\kappa$, '
                 r'pontilhada $=1-(2+C_1^N)/3$, cinza $=1/3$', fs=9)
    return fig


def fig3(plt):
    from matplotlib.ticker import NullFormatter
    d = _dados(3)
    wbs, gs, F = d['wbs'], d['gs'], d['F']
    T, sig, N = float(d['T']), float(d['sig']), int(d['N'])
    fig, ax = plt.subplots(figsize=FIGSIZE[3])
    pc = ax.pcolormesh(wbs, gs, np.log10(np.clip(1-F, 1e-6, 1)), shading='gouraud', cmap=CMAP)
    cs = ax.contour(wbs, gs, F, levels=NIVEIS_F, colors='w', linewidths=1)
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=7)
    W, G = np.meshgrid(wbs, gs); nb = 1/np.expm1(W/T)
    ax.contour(wbs, gs, N*(4*G)**2*(2*nb+1), levels=NIVEIS_X, colors='r', linestyles='--', linewidths=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\omega_b/\gamma$'); ax.set_ylabel(r'$g/\gamma$')
    plt.colorbar(pc, label=r'$\log_{10}(1-\bar F)$')
    if MOSTRAR_TITULO:
        ax.set_title(fr'Regime lento, $N={N}$; $k_BT={T:g}\hbar\gamma$, $\sigma={sig:g}\gamma$. '
                     r'Vermelho: $\frac{2}{3}N\kappa=0{,}01;\,0{,}1$', fontsize=8)
    return fig


def fig4(plt):
    d = _dados(4)
    sigs, Ns = d['sigs'], d['Ns']
    wb, g, k = float(d['wb']), float(d['g']), float(d['kappa'])
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE[4])
    for i, N in enumerate(Ns):
        c = CN[int(N)]; CN_ = C1_fechado(k)**N
        ax[0].plot(sigs, 1-d['Fopt'][i], c, marker='o', ms=MS, label=fr'$N={N}$')
        ax[0].axhline(1-(2+CN_)/3, color=c, ls='--', lw=0.8)
        ax[0].axhline(1-(1+CN_**2+CN_)/3, color=c, ls=':', lw=0.8)
        ax[1].plot(sigs, d['eta'][i], c, label=fr'$\eta$, $N={N}$')
        ax[1].plot(sigs, d['absC'][i], c+'--', lw=1)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$')
        a.axvline(wb, color='gray', lw=0.8, ls='-'); a.axvline(1, color='gray', lw=.5, ls=':')
        a.legend(fontsize=LEGENDA_FS)
    ax[0].set_yscale('log'); ax[0].set_ylabel(r'$1-\bar F$')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    _titulo(fig, fr'Cálculo exato; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$, $k_BT=0$ ($\kappa={k:.4f}$). '
                 r'Linha cinza: $\sigma=\omega_b$. Esq.: tracejada $=$ previsão $\sigma\gg\omega_b$, '
                 r'pontilhada $=$ previsão $\sigma\ll\omega_b$', fs=9)
    return fig


def fig5(plt):
    d = _dados(5)
    Ts, Ns, nbs, ks, F = d['Ts'], d['Ns'], d['nbs'], d['kappas'], d['Fopt']
    wb, g, sig = float(d['wb']), float(d['g']), float(d['sig'])
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE[5])
    for i, N in enumerate(Ns):
        c = CN[int(N)]
        ax[0].plot(Ts, 1-F[i], c, marker='o', ms=MS, label=fr'$N={N}$')
        ax[0].plot(Ts, (2/3)*N*ks, c+'--', lw=1)
        ax[1].plot(nbs, 1-F[i], c, marker='o', ms=MS, label=fr'$N={N}$')
        ax[1].plot(nbs, (2/3)*N*ks, c+'--', lw=1)
    ax[0].set_xlabel(r'$k_BT/\hbar\gamma$'); ax[1].set_xlabel(r'$\bar n$')
    for a in ax:
        a.set_yscale('log'); a.set_ylabel(r'$1-\bar F$  (tracejada: $\frac{2}{3}N\kappa$)')
        a.legend(fontsize=LEGENDA_FS)
    _titulo(fig, fr'Regime lento, na linha; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$, '
                 fr'$\sigma={sig:g}\gamma$')
    return fig


def fig6(plt):
    d = _dados(6)
    wbs, nbs, g, E = d['wbs'], d['nbs'], float(d['g']), d['perda']
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE[6])
    wf = np.geomspace(wbs[0], wbs[-1], 400)
    for i, (nb, c) in enumerate(zip(nbs, CORES_NB)):
        ax[0].plot(wbs, E[i], c+'o', ms=4, label=fr'exato, $\bar n={nb:g}$')
        ax[0].plot(wf, 32*g**2*(2*nb+1)/(1+4*wf**2), c, lw=1.2,
                   label=r'$32g^2(2\bar n+1)/(\gamma^2+4\omega_b^2)$' if i == 0 else None)
        ax[0].plot(wf, 2*(2*g/wf)**2*(2*nb+1), c+'--', lw=0.8,
                   label=r'rápido: $2S(2\bar n+1)$' if i == 0 else None)
        ax[0].axhline(32*g**2*(2*nb+1), color=c, ls=':', lw=0.8)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(3e-6, 3e-2)
    ax[0].axvline(0.5, color='gray', lw=0.5, ls=':')
    ax[0].set_xlabel(r'$\omega_b/\gamma$'); ax[0].set_ylabel(r'perda de coerência por sítio $1-\mathrm{Re}\,C_1$')
    if MOSTRAR_TITULO:
        ax[0].set_title(fr'$g={g:g}\gamma$; pontilhada: lento $2\kappa$', fontsize=9)
    ax[0].legend(fontsize=LEGENDA_FS)
    Ns = d['val_Ns']
    for j, (s, c) in enumerate(zip(d['val_sigs'], CORES_SIG)):
        ax[1].plot(Ns, 1-d['F_exato'][j], c+'o', ms=4, label=fr'exato, $\sigma={s:g}\gamma$')
        ax[1].plot(Ns, 1-d['F_estatico'][j], c, lw=1, label=fr'estático $(2+C)/3$, $\sigma={s:g}\gamma$')
        ax[1].plot(Ns, 1-d['F_CC'][j], c+'--', lw=1, label=fr'$(1+C^2+C)/3$, $\sigma={s:g}\gamma$')
    ax[1].set_xlabel(r'$N$'); ax[1].set_ylabel(r'$1-\bar F$')
    if MOSTRAR_TITULO:
        ax[1].set_title(fr'$\omega_b={float(d["val_wb"]):g}\gamma$, $g={float(d["val_g"]):g}\gamma$, $k_BT=0$',
                        fontsize=9)
    ax[1].legend(fontsize=LEGENDA_FS)
    return fig


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Plota as figuras do regime lento a partir de data/.")
    ap.add_argument('--figs', type=int, nargs='+', default=list(FIGS), choices=list(FIGS))
    ap.add_argument('--show', action='store_true', help="mostra as figuras na tela")
    a = ap.parse_args()
    plt = _plt(a.show)
    for k in a.figs:
        if not os.path.exists(os.path.join(PASTA, f'figL{k}.npz')):
            print(f"pulando L{k}: {PASTA}/figL{k}.npz nao existe (rode calcular_lento.py --figs {k})")
            continue
        fig = FIGS[k](plt)
        fig.tight_layout()
        _salvar(fig, k)
        if not a.show:
            plt.close(fig)
    if a.show:
        plt.show()
