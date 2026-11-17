"""Figuras da porta CZ de Brod-Combes (contrapropagante, chi -> inf) com osciladores no regime lento.

D1  Reproducao de BC Fig. 3(a) (kappa=0), N=1,3,5
D2  F x sigma estilo Fig. 3(a), 4 paineis kappa = 0, 1e-3, 1e-2, 5e-2
D3  F x sigma, um painel por N, varias kappa sobrepostas
D4  F_max e sigma_opt em funcao de kappa
D5  coeficiente c(sigma) = -(1/N) dF/dkappa|_0 (derivada exata por diferencas finitas)

Uso: python dois_fotons_figs.py --figs 1 2 3 4 5 --procs 2
"""
import argparse, os, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import qmc, norm
from dois_fotons import mus, F_de_mus, gam

ESTILO_N = {1: ('#A51C1C', 'o'), 3: ('#2FA82F', 'D'), 5: ('#2AB3BE', '^')}
KCOR = {0: 'k', 1e-3: '#1f77b4', 1e-2: '#2ca02c', 5e-2: '#d62728'}
NS = [1, 3, 5]
SIGS = np.logspace(-2.5, np.log10(0.6), 24)
KAPS = [0, 1e-3, 1e-2, 5e-2]
M = 128            # pontos de Sobol por media
L = 48


def _out(n):
    return os.path.join(os.getcwd(), n)


def _amostras(N, seed=7):
    s = qmc.Sobol(d=N, scramble=True, seed=seed).random(M)
    return norm.ppf(s)                       # (M, N) gaussianas padrao


def F_desordem(N, sig, kap):
    if kap == 0:
        D = [0.0] * N
        return F_de_mus(mus(D, D, sig, L=L))
    Z = _amostras(N)
    v = []
    for z in Z:
        D = list((gam / 2) * np.sqrt(kap) * z)     # Delta_j - Delta = 2 g x_j = (gamma/2) u_j
        v.append(F_de_mus(mus(D, D, sig, L=L)))
    return float(np.mean(v))


def _job(a):
    return F_desordem(*a)


def dFdk(N, sig, h=0.05):
    """dF/dkappa em kappa=0 = (1/2) sum_j d^2F/du_j^2 (u_j -> mesma dessintonia nas duas transicoes)."""
    D0 = [0.0] * N
    F0 = F_de_mus(mus(D0, D0, sig, L=L))
    s = 0.0
    for j in range(N):
        for sgn in (+1, -1):
            D = [0.0] * N; D[j] = sgn * (gam / 2) * h
            s += F_de_mus(mus(D, D, sig, L=L))
        s -= 2 * F0
    return 0.5 * s / h ** 2


def _jobd(a):
    return dFdk(*a)


T_INICIO = time.time()


def _hms(t):
    t = int(round(t))
    return f"{t // 3600:d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def _custo(a):
    """Custo relativo estimado de um job (o laco em mus e ~O(N^2) em arrays L^3)."""
    if len(a) == 3:                          # F_desordem(N, sig, kap): 1 ou M avaliacoes
        N, _, k = a
        return (1 if k == 0 else M) * N ** 2
    N = a[0]                                 # dFdk(N, sig): 2N+1 avaliacoes
    return (2 * N + 1) * N ** 2


def _barra(pool, fn, jobs, desc, larg=30):
    """Roda os jobs mostrando barra de progresso, tempo decorrido e restante (ponderado pelo custo)."""
    out = []; t0 = time.time()
    pesos = [_custo(j) for j in jobs]; total = sum(pesos); feito = 0
    for i, r in enumerate(pool.imap(fn, jobs, chunksize=1)):
        out.append(r)
        feito += pesos[i]; fr = feito / total
        el = time.time() - t0; rest = el * (1 - fr) / fr
        cheio = int(larg * fr)
        barra = '#' * cheio + '-' * (larg - cheio)
        print(f"\r{desc} [{barra}] {100 * fr:5.1f}%  {i + 1}/{len(jobs)}  "
              f"decorrido {_hms(el)}  faltam ~{_hms(rest)}  (total {_hms(time.time() - T_INICIO)})",
              end='', flush=True)
    print(f"\n{desc}: concluido em {_hms(time.time() - t0)}")
    return out


def calcular(procs):
    arq = _out('dois_fotons_dados.npz')
    if os.path.exists(arq):
        return dict(np.load(arq))
    jobs = [(N, s, k) for k in KAPS for N in NS for s in SIGS]
    with Pool(procs) as p:
        R = np.array(_barra(p, _job, jobs, '[1/3] F(sigma)')).reshape(len(KAPS), len(NS), len(SIGS))
        kgrid = np.array([0, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1])
        # F_max: maximiza em grade fina ao redor do otimo de kappa=0
        sfin = {N: SIGS[np.argmax(R[0, i])] * np.logspace(-0.25, 0.25, 7) for i, N in enumerate(NS)}
        jobs2 = [(N, s, k) for k in kgrid for N in NS for s in sfin[N]]
        R2 = np.array(_barra(p, _job, jobs2, '[2/3] F_max(kappa)')).reshape(len(kgrid), len(NS), 7)
        jobs3 = [(N, s) for N in NS for s in SIGS]
        D = np.array(_barra(p, _jobd, jobs3, '[3/3] dF/dkappa')).reshape(len(NS), len(SIGS))
    sf = np.array([sfin[N] for N in NS])
    dados = dict(R=R, kgrid=kgrid, R2=R2, sfin=sf, D=D)
    np.savez(arq, **dados)
    return dados


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
    return plt


def _moldura(a):
    a.tick_params(direction='in', which='both', top=True, right=True, labelsize=9)
    for sp in a.spines.values():
        sp.set_color('#888888')


def _curva(a, x, y, N, **extra):
    cor, mk = ESTILO_N[N]
    kw = dict(color=cor, lw=1.6)
    kw.update(extra)
    a.plot(x, y, **kw)


def _legN(fig, plt):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=ESTILO_N[N][0], lw=2.2) for N in NS]
    fig.legend(h, [str(N) for N in NS], title=r'$N$', loc='center right', frameon=False,
               fontsize=10, title_fontsize=12, handletextpad=0.3)


def fig1(d):
    plt = _plt()
    fig, a = plt.subplots(figsize=(5.6, 3.9))
    for i, N in enumerate(NS):
        y = d['R'][0, i]
        _curva(a, SIGS, y, N)
        j = np.argmax(y)
        a.annotate(fr'$F_{{\max}}={y[j]:.3f}$', (SIGS[j], y[j]), textcoords='offset points',
                   xytext={1: (-88, 0), 3: (8, 4), 5: (-72, 2)}[N], fontsize=8, color=ESTILO_N[N][0])
    a.axhline(0.4, color='#999999', ls='--', lw=0.8)
    a.text(0.0035, 0.375, r'$F=0.4$ (identidade $\times$ CZ)', fontsize=8, color='#666666')
    a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$'); a.set_ylabel(r'$\bar F$')
    a.set_ylim(0.35, 1.02); _moldura(a)
    a.set_title(r'Sem oscilador ($\kappa=0$): reprodução de BC, Fig. 3(a)', fontsize=10)
    _legN(fig, plt)
    plt.tight_layout(rect=(0, 0, 0.9, 1)); plt.savefig(_out('figD1_BC_kappa0.png'), dpi=150); plt.close(fig)


def fig2(d):
    plt = _plt()
    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True, sharey=True)
    for ik, k in enumerate(KAPS):
        a = ax.flat[ik]
        for i, N in enumerate(NS):
            if k > 0:
                a.plot(SIGS, d['R'][0, i], color=ESTILO_N[N][0], lw=0.8, ls='--', alpha=0.6)
            _curva(a, SIGS, d['R'][ik, i], N)
            a.text(0.04, 0.82 - 0.075 * i, fr'$N={N}$: $F_{{\max}}={d["R"][ik, i].max():.3f}$',
                   transform=a.transAxes, fontsize=8.5, color=ESTILO_N[N][0])
        a.set_xscale('log'); a.set_ylim(0.35, 1.0); _moldura(a)
        a.text(0.04, 0.92, (r'$\kappa=0$ (BC)' if k == 0 else fr'$\kappa={k:g}$'),
               transform=a.transAxes, fontsize=10)
        a.axhline(0.4, color='#BBBBBB', lw=0.7)
    for a in ax[1]:
        a.set_xlabel(r'$\sigma/\gamma$')
    for a in ax[:, 0]:
        a.set_ylabel(r'$\bar F$')
    _legN(fig, plt)
    fig.suptitle(r'CZ contrapropagante com osciladores lentos (tracejado: $\kappa=0$)', fontsize=10)
    plt.tight_layout(rect=(0, 0, 0.92, 1)); plt.savefig(_out('figD2_F_sigma_kappas.png'), dpi=150); plt.close(fig)


def fig3(d):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for i, N in enumerate(NS):
        a = ax[i]
        for ik, k in enumerate(KAPS):
            a.plot(SIGS, d['R'][ik, i], color=KCOR[k], lw=1.6,
                   label=(r'$0$' if k == 0 else f'{k:g}'))
            a.text(0.04, 0.92 - 0.075 * ik,
                   (r'$\kappa=0$' if k == 0 else fr'$\kappa={k:g}$') + fr': $F_{{\max}}={d["R"][ik, i].max():.3f}$',
                   transform=a.transAxes, fontsize=8, color=KCOR[k])
        a.set_xscale('log'); a.set_title(fr'$N={N}$', fontsize=10); a.set_xlabel(r'$\sigma/\gamma$')
        a.set_ylim(0.35, 1.0); _moldura(a)
    ax[0].set_ylabel(r'$\bar F$')
    ax[2].legend(title=r'$\kappa$', fontsize=8, frameon=False, loc='lower right')
    plt.tight_layout(); plt.savefig(_out('figD3_por_N.png'), dpi=150); plt.close(fig)


def fig4(d):
    plt = _plt()
    kg, R2, sf = d['kgrid'], d['R2'], d['sfin']
    fig, ax0 = plt.subplots(figsize=(6.0, 4.0)); ax = [ax0]
    D = d['D']
    for i, N in enumerate(NS):
        Fm = R2[:, i].max(axis=1); so = sf[i][R2[:, i].argmax(axis=1)]
        j0 = np.argmax(R2[0, i]); s0 = sf[i][j0]
        c = -np.interp(np.log(s0), np.log(SIGS), D[i])          # -dF/dkappa no sigma otimo
        x = kg[1:]
        _curva(ax[0], x, 1 - Fm[1:], N, ls='-')
        ax[0].plot(x, (1 - Fm[0]) + c * x, color=ESTILO_N[N][0], ls='--', lw=0.9)
        ax[0].axhline(1 - Fm[0], color=ESTILO_N[N][0], ls=':', lw=0.8)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\kappa$'); ax[0].set_ylabel(r'$1-\bar F_{\max}$')
    ax[0].set_title(r'tracejado: $1-\bar F_{\max}^{(0)}+c\,\kappa$; pontilhado: $\kappa=0$', fontsize=9)
    for a in ax:
        _moldura(a)
    _legN(fig, plt)
    plt.tight_layout(rect=(0, 0, 0.9, 1)); plt.savefig(_out('figD4_Fmax_kappa.png'), dpi=150); plt.close(fig)


def fig5(d):
    plt = _plt()
    fig, a = plt.subplots(figsize=(5.6, 3.7))
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
    _moldura(a); _legN(fig, plt)
    plt.tight_layout(rect=(0, 0, 0.9, 1)); plt.savefig(_out('figD5_coef.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--procs', type=int, default=os.cpu_count())
    A = ap.parse_args()
    d = calcular(A.procs)
    for k in A.figs:
        t = time.time()
        globals()[f'fig{k}'](d)
        print(f"figura D{k}: {time.time() - t:.1f}s")
    print(f"ok -- tempo total {_hms(time.time() - T_INICIO)}")
