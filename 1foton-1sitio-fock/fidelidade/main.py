"""Qubit dual-rail de 1 foton, 1 sitio de Brod-Combes, oscilador em ESTADO DE FOCK |n>.

Mesma analise do caso termico (core.py), trocando p_n termico por p_n = delta_{n,n0}.
Requer core.py na mesma pasta deste arquivo.

Uso:
    python main.py              # gera as 5 figuras (.svg e .png) no diretorio corrente
    python main.py --figs 1 5   # so as figuras 1 e 5
    python main.py --procs 4    # numero de processos
Mostra uma barra por figura e uma barra global com o tempo total restante estimado.
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from core import quantities, resonant, nbar_of, _mapa_paralelo, _fmt_t, thermal_p  # noqa: E402


def _out(nome):
    caminho = os.path.join(os.getcwd(), nome)
    print(f"salvo: {caminho}")
    return caminho


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Estilo LaTeX: mathtext com Computer Modern (mesmo estilo de plot_fock.py).
    # Nao usa text.usetex=True porque o PNG exigiria o dvipng, que nao esta instalado.
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.0,
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
        'svg.fonttype': 'path',
        'lines.linewidth': 1.2,
    })
    return plt


def _salvar(plt, fig, nome):
    """Salva a figura em .svg e .png (300 dpi)."""
    fig.tight_layout()
    fig.savefig(_out(nome + '.svg'), dpi=300, bbox_inches='tight')
    fig.savefig(_out(nome + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


CORES = {0: 'k', 1: 'b', 2: 'r', 3: 'g'}


# ---------- Figura 1F: varredura em frequencia ----------
F1_DELTA, F1_WB, F1_G, F1_SIG = 0.0, 3.0, 0.75, 0.1
F1_NUS = np.linspace(-5, 5, 401)
F1_NS = [0, 1, 2]


def _run1(args):
    n, nu = args
    return quantities(nu, F1_SIG, F1_DELTA, F1_WB, F1_G, 0.0, Ngrid=601, fock=n)


def fig1(procs):
    plt = _plt()
    Delta, wb, g, sig = F1_DELTA, F1_WB, F1_G, F1_SIG
    S = (2*g/wb)**2
    nus, ns = F1_NUS, F1_NS
    pontos = [(n, nu) for n in ns for nu in nus]
    todos = _mapa_paralelo(_run1, pontos, procs, "Figura 1F")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    for i, n in enumerate(ns):
        R = todos[i*len(nus):(i+1)*len(nus)]
        c = CORES[n]
        eta = np.array([r['eta'] for r in R]); C = np.array([r['C'] for r in R])
        F = np.array([r['Fbar'] for r in R]); Fo = np.array([r['Fopt'] for r in R])
        lab = fr'Fock $|n={n}\rangle$'
        ax[0, 0].plot(nus, eta, c, label=lab)
        ax[0, 1].plot(nus, abs(C), c, label=lab)
        ax[1, 0].plot(nus, F, c, label=lab); ax[1, 0].plot(nus, Fo, c+'--', lw=1.2)
        ax[1, 1].plot(nus, np.angle(C)/np.pi, c, label=fr'$n={n}$')
        if n == 0:
            ph = np.array([r['phase'] for r in R])
            ax2 = ax[1, 1].twinx()
            ax2.plot(nus, np.mod(ph, 2*np.pi)/np.pi, color='gray', lw=2.5, alpha=0.35)
            ax2.set_ylabel(r'fase ideal $\arg t_0/\pi$ (cinza)', color='gray')
            ax2.tick_params(axis='y', colors='gray'); ax2.set_ylim(-0.05, 2.05)
    for a in ax.flat:
        for k in range(-2, 3):
            a.axvline(Delta - 4*g**2/wb + k*wb, color='gray', lw=0.5, ls=':')
        a.set_xlabel(r'$\omega_c/\gamma$')
    ax[0, 0].set_ylabel(r'$\eta$ (população no modo ideal)')
    ax[0, 1].set_ylabel(r'$|C|$ (coerência)')
    ax[1, 0].set_ylabel(r'$\bar F$ (cheia) e $\bar F$ com correção de fase (tracejada)')
    ax[1, 1].set_ylabel(r'erro de fase $\arg C/\pi$ (coloridas)')
    for a in ax.flat:
        a.legend(fontsize=8)
    fig.suptitle(fr'Banho em Fock; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$, $S={S:.2f}$, $\sigma={sig:g}\gamma$',
                 fontsize=10)
    _salvar(plt, fig, 'figF1_vs_freq')


# ---------- Figura 2F: frequencia do oscilador ----------
F2_SIG, F2_G = 0.1, 0.5
F2_WBS = np.geomspace(0.5, 30, 22)
F2_NS = [0, 1, 2]


def _run2(args):
    wb, n, tipo = args
    if tipo == 'fock':
        return resonant(F2_SIG, wb, F2_G, 0.0, Ngrid=601, fock=n)
    return resonant(F2_SIG, wb, F2_G, float(n), Ngrid=601)


def fig2(procs):
    plt = _plt()
    sig, g, wbs, ns = F2_SIG, F2_G, F2_WBS, F2_NS
    jobs = [(wb, n, t) for t in ('fock', 'term') for n in ns for wb in wbs]
    R = _mapa_paralelo(_run2, jobs, procs, "Figura 2F")
    R = np.array(R, dtype=object).reshape(2, len(ns), len(wbs))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    S = (2*g/wbs)**2
    for i, n in enumerate(ns):
        c = CORES[n]
        Ff = np.array([r['Fopt'] for r in R[0, i]]); Ft = np.array([r['Fopt'] for r in R[1, i]])
        eta = np.array([r['eta'] for r in R[0, i]]); C = np.array([abs(r['C']) for r in R[0, i]])
        ax[0].plot(wbs, 1-Ff, c, label=fr'Fock $n={n}$')
        ax[0].plot(wbs, 1-Ft, c+':', lw=1.2, label=fr'térmico $\bar n={n}$' if n > 0 else None)
        ax[0].plot(wbs, 2*S*(2*n+1), c+'--', lw=1.2)
        ax[1].plot(wbs, eta, c, label=fr'$\eta$, $n={n}$'); ax[1].plot(wbs, C, c+'--', label=fr'$|C|$, $n={n}$')
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[1].set_xscale('log')
    for a in ax:
        a.axvline(1.0, color='gray', lw=0.5, ls=':'); a.set_xlabel(r'$\omega_b/\gamma$')
    ax[0].set_ylabel(r'$1-\bar F$  (tracejada: $2S(2n+1)$)')
    ax[1].set_ylabel(r'população $\eta$ (cheia), coerência $|C|$ (tracejada)')
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.suptitle(fr'Banho em Fock, ressonante (fase coerente $=\pi$); $g={g:g}\gamma$, $\sigma={sig:g}\gamma$',
                 fontsize=10)
    _salvar(plt, fig, 'figF2_vs_wb')


# ---------- Figura 3F: mapa (omega_b, g) com Fock n fixo ----------
F3_SIG, F3_N = 0.1, 1
F3_WBS = np.geomspace(0.5, 30, 14)
F3_GS = np.geomspace(0.05, 1.5, 14)


def _run3(args):
    wb, g = args
    return resonant(F3_SIG, wb, g, 0.0, Ngrid=401, fock=F3_N)['Fopt']


def fig3(procs):
    plt = _plt()
    sig, n, wbs, gs = F3_SIG, F3_N, F3_WBS, F3_GS
    jobs = [(wb, g) for g in gs for wb in wbs]
    F = np.array(_mapa_paralelo(_run3, jobs, procs, "Figura 3F")).reshape(len(gs), len(wbs))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pc = ax.pcolormesh(wbs, gs, np.log10(np.clip(1-F, 1e-5, 1)), shading='gouraud', cmap='viridis')
    cs = ax.contour(wbs, gs, F, levels=[0.9, 0.99, 0.999], colors='w', linewidths=1.2)
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=8)
    W, G = np.meshgrid(wbs, gs)
    ax.contour(wbs, gs, (2*G/W)**2*(2*n+1), levels=[0.01, 0.1], colors='r', linestyles='--', linewidths=1.2)
    ax.axvline(1.0, color='gray', lw=0.5, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\omega_b/\gamma$'); ax.set_ylabel(r'$g/\gamma$')
    plt.colorbar(pc, label=r'$\log_{10}(1-\bar F)$')
    ax.set_title(fr'Fock $|n={n}\rangle$, ressonante; $\sigma={sig:g}\gamma$. '
                 r'Vermelho: $S(2n+1)=0{,}01;\,0{,}1$', fontsize=9)
    _salvar(plt, fig, 'figF3_mapa')


# ---------- Figura 4F: largura do pacote ----------
F4_N = 1
F4_SIGS = np.geomspace(0.01, 3, 18)
F4_CASES = [(10.0, 0.5, 'k'), (3.0, 0.3, 'b'), (0.5, 0.1, 'r')]


def _run4(a):
    sig, wb, g = a
    r = resonant(sig, wb, g, 0.0, Ngrid=801, fock=F4_N)
    return (r['Fopt'], r['eta'], abs(r['C']))


def fig4(procs):
    plt = _plt()
    n, sigs, cases = F4_N, F4_SIGS, F4_CASES
    jobs = [(s, wb, g) for wb, g, _ in cases for s in sigs]
    R = np.array(_mapa_paralelo(_run4, jobs, procs, "Figura 4F")).reshape(len(cases), len(sigs), 3)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, (wb, g, c) in enumerate(cases):
        S = (2*g/wb)**2
        lab = fr'$\omega_b={wb:g}\gamma$, $S={S:.2f}$'
        ax[0].plot(sigs, 1-R[i, :, 0], c, label=lab)
        ax[1].plot(sigs, R[i, :, 1], c, label=r'$\eta$, '+lab)
        ax[1].plot(sigs, R[i, :, 2], c+'--', lw=1.2)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$'); a.axvline(1, color='gray', lw=.5, ls=':')
    ax[0].set_yscale('log'); ax[0].set_ylabel(r'$1-\bar F$')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=7)
    fig.suptitle(fr'Banho em Fock $|n={n}\rangle$, ressonante (fase coerente $=\pi$)', fontsize=10)
    _salvar(plt, fig, 'figF4_vs_sigma')


# ---------- Figura 5F: Fock x termico em funcao de S (oscilador rapido) ----------
F5_WB, F5_SIG = 10.0, 0.1
F5_S = np.geomspace(1e-3, 3.0, 40)
F5_NS = [0, 1, 2, 3]


def _run5(a):
    S, n, tipo = a
    g = 0.5*F5_WB*np.sqrt(S)
    c = -4*g**2/F5_WB                      # linha deslocada pelo polaron
    if tipo == 'fock':
        r = quantities(c, F5_SIG, 0.0, F5_WB, g, 0.0, Ngrid=601, Dref=c, fock=n)
    else:
        r = quantities(c, F5_SIG, 0.0, F5_WB, g, float(n), Ngrid=601, Dref=c)
    return (r['Fbar'], r['Fopt'], r['eta'], r['C'])


def fig5(procs):
    from scipy.special import eval_laguerre
    plt = _plt()
    Ss, ns = F5_S, F5_NS
    jobs = [(S, n, t) for t in ('fock', 'term') for n in ns for S in Ss]
    R = np.array(_mapa_paralelo(_run5, jobs, procs, "Figura 5F"), dtype=complex).reshape(2, len(ns), len(Ss), 4)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, n in enumerate(ns):
        c = CORES[n]
        ax[0].plot(Ss, R[0, i, :, 0].real, c, label=fr'Fock $n={n}$')
        ax[1].plot(Ss, np.exp(-Ss)*eval_laguerre(n, Ss)**2, c, label=fr'Fock: $e^{{-S}}L_{n}(S)^2$')
        if n > 0:
            ax[0].plot(Ss, R[1, i, :, 0].real, c+'--', lw=1.2, label=fr'térmico $\bar n={n}$')
            pk = thermal_p(float(n), 150); pk = pk/pk.sum()
            zpl = sum(pk[k]*np.exp(-Ss)*eval_laguerre(k, Ss)**2 for k in range(151))
            ax[1].plot(Ss, zpl, c+'--', lw=1.2, label=fr'térmico $\bar n={n}$')
    ax[0].set_ylabel(r'$\bar F$')
    ax[1].set_ylabel(r'peso da linha de zero fônon $w$')
    ax[1].axhline(0.5, color='gray', lw=0.5, ls=':')
    for a in ax:
        a.set_xscale('log'); a.set_xlabel(r'$S=(2g/\omega_b)^2$')
        a.legend(fontsize=7)
    fig.suptitle(fr'Fock (cheia) $\times$ térmico com mesma ocupação média (tracejada); '
                 fr'$\omega_b={F5_WB:g}\gamma$, $\sigma={F5_SIG:g}\gamma$, fóton na linha do pólaron', fontsize=10)
    _salvar(plt, fig, 'figF5_fock_vs_termico')


def _n_pontos(k):
    """Numero de pontos calculados em cada figura (para a barra global)."""
    return {1: len(F1_NS) * len(F1_NUS),
            2: 2 * len(F2_NS) * len(F2_WBS),
            3: len(F3_GS) * len(F3_WBS),
            4: len(F4_CASES) * len(F4_SIGS),
            5: 2 * len(F5_NS) * len(F5_S)}[k]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Figuras do qubit dual-rail com banho em estado de Fock.")
    ap.add_argument('--figs', type=int, nargs='+', default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    print(f"diretorio de saida: {os.getcwd()}")
    t_total = time.time()
    try:
        from tqdm import tqdm
        # barra global: avanca a cada ponto de qualquer figura -> tempo total restante
        core._BARRA_TOTAL = tqdm(total=sum(_n_pontos(k) for k in a.figs), desc="Total", position=0,
                                 dynamic_ncols=True,
                                 bar_format="{desc}: {percentage:5.1f}%|{bar:30}| {n}/{total} pontos "
                                            "[decorrido {elapsed}, restam ~{remaining}]")
    except ImportError:
        pass
    for k in a.figs:
        {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}[k](a.procs)
    if core._BARRA_TOTAL is not None:
        core._BARRA_TOTAL.close()
    print(f"Total: {_fmt_t(time.time() - t_total)}")
