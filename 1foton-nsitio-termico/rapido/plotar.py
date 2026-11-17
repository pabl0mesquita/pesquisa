"""Plota as figuras N1-N5 a partir de ./data/figN{k}.npz (gerados por calcular.py).
Nao faz nenhum calculo pesado: ajuste livremente o estilo na secao ESTILO abaixo.

Uso:
    python plotar.py                # todas as figuras com dados disponiveis
    python plotar.py --figs 2 5     # so algumas
    python plotar.py --show         # abre as janelas alem de salvar
"""
import os
import glob
import argparse
import numpy as np

# =====================================================================
#  ESTILO (edite aqui)
# =====================================================================
PASTA = 'data'                 # onde estao os .npz (relativa ao diretorio corrente)
FORMATOS = [                   # ex.: 'png', 'svg', 'pdf'
    # 'svg',
    'png',
]
DPI = 300
MOSTRAR_TITULO = True          # suptitle/title com os parametros

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

CN = {1: 'k', 2: 'b', 5: 'r', 10: 'g', 20: 'm'}   # cor por numero de sitios N
CORES_G = ['k', 'b', 'r', 'g']                      # cores das curvas de g (fig. N2)
LEGENDA_FS = 7
TITULO_FS = 10
MS = 2.5                                            # tamanho dos marcadores

FIGSIZE = {1: (10, 7.2), 2: (10, 3.9), 3: (6.4, 4.8), 4: (10, 3.9), 5: (10, 3.9)}
ARQUIVO = {1: 'figN1_vs_freq', 2: 'figN2_vs_N', 3: 'figN3_mapa_N10',
           4: 'figN4_vs_sigma', 5: 'figN5_vs_T'}

# fig. N3
CMAP = 'viridis'
NIVEIS_F = [0.9, 0.99, 0.999]       # contornos brancos de F
NIVEIS_X = [0.01, 0.1]              # contornos vermelhos de N S (2 nbar + 1)


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
    return np.load(os.path.join(PASTA, f'figN{k}.npz'))


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
    wb, g, sig, T, nb, S, Dref = (float(d[k]) for k in ('wb', 'g', 'sig', 'T', 'nb', 'S', 'Dref'))
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
    for a in ax.flat:
        for k in range(-1, 2):
            a.axvline(Dref + k*wb, color='gray', lw=0.5, ls=':')
        a.set_xlabel(r'$\omega_c/\gamma$'); a.legend(fontsize=LEGENDA_FS)
    ax[0, 0].set_ylabel(r'$\eta$ (população no modo ideal)')
    ax[0, 1].set_ylabel(r'$|C|$ (coerência)')
    ax[1, 0].set_ylabel(r'$\bar F$ (cheia) e com correção de fase (tracejada)')
    ax[1, 1].set_ylabel(r'erro de fase $\arg C/\pi$')
    _titulo(fig, fr'$N$ sítios; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$, $S={S:.2f}$, '
                 fr'$k_BT={T:g}\hbar\gamma$ ($\bar n={nb:.3f}$), $\sigma={sig:g}\gamma$')
    return fig


def fig2(plt):
    d = _dados(2)
    Ns, gs = d['Ns'], d['gs']
    wb, sig, T, nb = (float(d[k]) for k in ('wb', 'sig', 'T', 'nb'))
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE[2])
    for i, (g, c) in enumerate(zip(gs, CORES_G)):
        S = (2*g/wb)**2; x = S*(2*nb+1)
        ax[0].plot(Ns, 1-d['Fopt'][i], c, marker='o', ms=MS, label=fr'$S={S:.3f}$')
        ax[0].plot(Ns, 2*Ns*x, c+'--', lw=1)
        ax[0].plot(Ns, 1-(1+(1-4*x)**Ns+(1-2*x)**Ns)/3, c+':', lw=1.2)
        ax[1].plot(Ns, d['eta'][i], c, label=fr'$\eta$, $S={S:.3f}$')
        ax[1].plot(Ns, d['absC'][i], c+'--', lw=1)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$N$'); ax[1].set_xlabel(r'$N$')
    ax[0].set_ylabel(r'$1-\bar F$')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    ax[0].legend(fontsize=LEGENDA_FS); ax[1].legend(fontsize=LEGENDA_FS)
    _titulo(fig, fr'Ressonante; $\omega_b={wb:g}\gamma$, $k_BT={T:g}\hbar\gamma$ ($\bar n={nb:.3f}$), $\sigma={sig:g}\gamma$. '
                 r'Esq.: cheia = numérico, tracejada = $2NS(2\bar n+1)$, pontilhada = modelo exponencial', fs=9)
    return fig


def fig3(plt):
    d = _dados(3)
    wbs, gs, F = d['wbs'], d['gs'], d['F']
    T, sig, N = float(d['T']), float(d['sig']), int(d['N'])
    fig, ax = plt.subplots(figsize=FIGSIZE[3])
    pc = ax.pcolormesh(wbs, gs, np.log10(np.clip(1-F, 1e-5, 1)), shading='gouraud', cmap=CMAP)
    cs = ax.contour(wbs, gs, F, levels=NIVEIS_F, colors='w', linewidths=1)
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=7)
    W, G = np.meshgrid(wbs, gs); nb = 1/np.expm1(W/T)
    ax.contour(wbs, gs, N*(2*G/W)**2*(2*nb+1), levels=NIVEIS_X, colors='r', linestyles='--', linewidths=1)
    ax.axvline(1.0, color='gray', lw=0.5, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\omega_b/\gamma$'); ax.set_ylabel(r'$g/\gamma$')
    plt.colorbar(pc, label=r'$\log_{10}(1-\bar F)$')
    if MOSTRAR_TITULO:
        ax.set_title(fr'$N={N}$ sítios, ressonante; $k_BT={T:g}\hbar\gamma$, $\sigma={sig:g}\gamma$. '
                     r'Vermelho: $NS(2\bar n+1)=0{,}01;\,0{,}1$', fontsize=8)
    return fig


def fig4(plt):
    d = _dados(4)
    sigs, Ns = d['sigs'], d['Ns']
    wb, g, T, nb, S = (float(d[k]) for k in ('wb', 'g', 'T', 'nb', 'S'))
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE[4])
    for i, N in enumerate(Ns):
        c = CN[int(N)]
        ax[0].plot(sigs, 1-d['Fopt'][i], c, label=fr'$N={N}$')
        ax[0].axhline(2*N*S*(2*nb+1), color=c, ls='--', lw=0.8)
        ax[1].plot(sigs, d['eta'][i], c, label=fr'$\eta$, $N={N}$')
        ax[1].plot(sigs, d['absC'][i], c+'--', lw=1)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$'); a.axvline(1, color='gray', lw=.5, ls=':')
        a.legend(fontsize=LEGENDA_FS)
    ax[0].set_yscale('log'); ax[0].set_ylabel(r'$1-\bar F$  (tracejada: $2NS(2\bar n+1)$)')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    _titulo(fig, fr'Ressonante; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$ ($S={S:.2f}$), '
                 fr'$k_BT={T:g}\hbar\gamma$')
    return fig


def fig5(plt):
    d = _dados(5)
    Ts, Ns, nbs, F = d['Ts'], d['Ns'], d['nbs'], d['Fopt']
    wb, g, sig, S = (float(d[k]) for k in ('wb', 'g', 'sig', 'S'))
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE[5])
    for i, N in enumerate(Ns):
        c = CN[int(N)]
        ax[0].plot(Ts, 1-F[i], c, marker='o', ms=MS, label=fr'$N={N}$')
        ax[0].plot(Ts, 2*N*S*(2*nbs+1), c+'--', lw=1)
        ax[1].plot(nbs, 1-F[i], c, marker='o', ms=MS, label=fr'$N={N}$')
        ax[1].plot(nbs, 2*N*S*(2*nbs+1), c+'--', lw=1)
    ax[0].set_xlabel(r'$k_BT/\hbar\gamma$'); ax[1].set_xlabel(r'$\bar n$')
    for a in ax:
        a.set_yscale('log'); a.set_ylabel(r'$1-\bar F$  (tracejada: $2NS(2\bar n+1)$)')
        a.legend(fontsize=LEGENDA_FS)
    _titulo(fig, fr'Ressonante; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$ ($S={S:.4f}$), '
                 fr'$\sigma={sig:g}\gamma$')
    return fig


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Plota as figuras N1-N5 a partir de data/.")
    ap.add_argument('--figs', type=int, nargs='+', default=list(FIGS), choices=list(FIGS))
    ap.add_argument('--show', action='store_true', help="mostra as figuras na tela")
    a = ap.parse_args()
    plt = _plt(a.show)
    for k in a.figs:
        if not os.path.exists(os.path.join(PASTA, f'figN{k}.npz')):
            print(f"pulando N{k}: {PASTA}/figN{k}.npz nao existe (rode calcular.py --figs {k})")
            continue
        fig = FIGS[k](plt)
        fig.tight_layout()
        _salvar(fig, k)
        if not a.show:
            plt.close(fig)
    if a.show:
        plt.show()
