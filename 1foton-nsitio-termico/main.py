"""Figuras do qubit dual-rail de 1 foton em N sitios com osciladores termicos.
Salva no diretorio corrente. Requer nsitios.py na mesma pasta.

Uso:
    python main.py                # todas as figuras
    python main.py --figs 2 5     # so algumas
    python main.py --procs 4
    python main.py --latex on     # forca LaTeX real (text.usetex; PNG requer dvipng)
Mostra uma barra por figura (pontos, %, decorrido, restante) e uma barra global
com o tempo total restante estimado.
"""
import os
import sys
import time
import shutil
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsitios import cadeia, cadeia_ressonante  # noqa: E402

def nbar_of(wb, T):
    """Ocupacao termica media do oscilador (antes importada de core.py)."""
    return 0.0 if T == 0 else 1.0 / np.expm1(wb / T)


CN = {1: 'k', 2: 'b', 5: 'r', 10: 'g', 20: 'm'}


def _out(nome):
    caminho = os.path.join(os.getcwd(), nome)
    (_G.barra.write if _G.barra is not None else print)(f"salvo: {caminho}")
    return caminho


# =====================================================================
#  Estilo LaTeX
# =====================================================================
LATEX = 'auto'   # 'auto' | 'on' | 'off'  (definido por --latex)


def _usa_usetex():
    """LaTeX real precisa de latex + dvipng (para PNG). Em 'auto' so ativa se ambos existirem."""
    tem = shutil.which('latex') is not None and shutil.which('dvipng') is not None
    if LATEX == 'on':
        if not tem:
            print("aviso: --latex on, mas latex/dvipng nao encontrados; o PNG pode falhar.")
        return True
    return LATEX == 'auto' and tem


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',          # fallback sem LaTeX: Computer Modern no mathtext
        'mathtext.rm': 'serif',
        'axes.linewidth': 1.0,
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
        'svg.fonttype': 'path',
        'lines.linewidth': 1.2,
    })
    if _usa_usetex():
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage[utf8]{inputenc}\usepackage{amsmath}\usepackage{amssymb}',
        })
    else:
        plt.rcParams['text.usetex'] = False
    return plt


# =====================================================================
#  Progresso: barra por figura + barra global com tempo restante
# =====================================================================
def _fmt_t(s):
    s = int(round(max(s, 0)))
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


class _Global:
    """Contador global de pontos (todas as figuras selecionadas)."""
    def __init__(self):
        self.total, self.n, self.t0, self.barra = 0, 0, time.time(), None


_G = _Global()
_FMT = ("{desc}: {percentage:5.1f}%|{bar:30}| {n}/{total} pontos "
        "[decorrido {elapsed}, restam ~{remaining}]")


def _iniciar_global(total):
    _G.total, _G.n, _G.t0 = total, 0, time.time()
    try:
        from tqdm import tqdm
        _G.barra = tqdm(total=total, desc="Total    ", position=0, dynamic_ncols=True, bar_format=_FMT)
    except ImportError:
        _G.barra = None


def _fechar_global():
    if _G.barra is not None:
        _G.barra.close()


def _progresso(iteravel, total, desc):
    """Itera mostrando, para a figura atual, % concluido, decorrido e restante;
    e atualiza a barra global (tempo restante de todas as figuras)."""
    t0 = time.time()
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    if tqdm is not None:
        with tqdm(total=total, desc=desc, position=1, leave=False, dynamic_ncols=True,
                  bar_format=_FMT) as b:
            for x in iteravel:
                yield x
                b.update()
                _G.n += 1
                if _G.barra is not None:
                    _G.barra.update()
        (_G.barra.write if _G.barra is not None else print)(
            f"{desc}: {total} pontos em {_fmt_t(time.time() - t0)}")
        return
    # fallback sem dependencias: uma linha com figura + total
    for i, x in enumerate(iteravel, 1):
        yield x
        _G.n += 1
        dt = time.time() - t0; dtg = time.time() - _G.t0
        resta = dt * (total - i) / i
        resta_g = dtg * (_G.total - _G.n) / _G.n if _G.total else 0
        cheio = int(30 * i / total)
        sys.stdout.write(f"\r{desc}: [{'#'*cheio}{'-'*(30-cheio)}] {i}/{total} ({100*i/total:5.1f}%) "
                         f"decorrido {_fmt_t(dt)}, restam ~{_fmt_t(resta)} | "
                         f"total {100*_G.n/max(_G.total, 1):5.1f}%, restam ~{_fmt_t(resta_g)}   ")
        sys.stdout.flush()
    sys.stdout.write(f"\n{desc}: {total} pontos em {_fmt_t(time.time() - t0)}\n")


def _mapa_paralelo(func, jobs, procs, desc):
    """Pool.imap (mantem a ordem) com barra de progresso."""
    with Pool(procs) as P:
        return list(_progresso(P.imap(func, jobs, chunksize=1), len(jobs), desc))


# ---------- Figura N1: varredura em frequencia ----------
N1 = dict(wb=10.0, g=0.5, sig=0.1, T=5.0, Ns=[1, 2, 5, 10])


def _run1(a):
    nu, N = a
    P = N1; nb = nbar_of(P['wb'], P['T']); Dref = -4*P['g']**2/P['wb']
    return cadeia(nu, P['sig'], N, P['wb'], P['g'], nb, Dref=Dref)


def fig1(procs):
    plt = _plt()
    P = N1; wb, g, sig, T, Ns = P['wb'], P['g'], P['sig'], P['T'], P['Ns']
    nb = nbar_of(wb, T); S = (2*g/wb)**2; Dref = -4*g**2/wb
    nus = np.concatenate([np.linspace(-13, -2, 120), np.linspace(-2, 2, 241), np.linspace(2, 13, 120)])
    nus = np.unique(nus)
    jobs = [(nu, N) for N in Ns for nu in nus]
    R = _mapa_paralelo(_run1, jobs, procs, "Figura N1")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    for i, N in enumerate(Ns):
        r = R[i*len(nus):(i+1)*len(nus)]
        c = CN[N]; lab = fr'$N={N}$'
        eta = np.array([q['eta'] for q in r]); C = np.array([q['C'] for q in r])
        F = np.array([q['Fbar'] for q in r]); Fo = np.array([q['Fopt'] for q in r])
        ax[0, 0].plot(nus, eta, c, label=lab)
        ax[0, 1].plot(nus, abs(C), c, label=lab)
        ax[1, 0].plot(nus, F, c, label=lab); ax[1, 0].plot(nus, Fo, c+'--', lw=1)
        ax[1, 1].plot(nus, np.angle(C)/np.pi, c, label=fr'$\arg C$, $N={N}$')
    ph1 = np.array([q['phase'] for q in R[:len(nus)]])
    ax2 = ax[1, 1].twinx()
    ax2.plot(nus, np.mod(ph1, 2*np.pi)/np.pi, color='gray', lw=2.5, alpha=0.35)
    ax2.set_ylabel(r'fase ideal de 1 sítio $\arg t_0/\pi$ (cinza)', color='gray')
    ax2.tick_params(axis='y', colors='gray'); ax2.set_ylim(-0.05, 2.05)
    ax[1, 1].set_ylim(-1.05, 1.05)
    for a in ax.flat:
        for k in range(-1, 2):
            a.axvline(Dref + k*wb, color='gray', lw=0.5, ls=':')
        a.set_xlabel(r'$\omega_c/\gamma$'); a.legend(fontsize=7)
    ax[0, 0].set_ylabel(r'$\eta$ (população no modo ideal)')
    ax[0, 1].set_ylabel(r'$|C|$ (coerência)')
    ax[1, 0].set_ylabel(r'$\bar F$ (cheia) e com correção de fase (tracejada)')
    ax[1, 1].set_ylabel(r'erro de fase $\arg C/\pi$')
    fig.suptitle(fr'$N$ sítios; $\omega_b={wb:g}\gamma$, $g={g:g}\gamma$, $S={S:.2f}$, '
                 fr'$k_BT={T:g}\hbar\gamma$ ($\bar n={nb:.3f}$), $\sigma={sig:g}\gamma$', fontsize=10)
    plt.tight_layout(); plt.savefig(_out('figN1_vs_freq.png'), dpi=150); plt.close(fig)


# ---------- Figura N2: acumulo com N ----------
N2 = dict(wb=10.0, sig=0.1, T=5.0, gs=[0.158, 0.316, 0.5, 1.0], Ns=list(range(1, 31)))


def _run2(a):
    N, g = a
    P = N2
    return cadeia_ressonante(P['sig'], N, P['wb'], g, nbar_of(P['wb'], P['T']))


def fig2(procs):
    plt = _plt()
    P = N2; wb, T, gs, Ns = P['wb'], P['T'], P['gs'], np.array(P['Ns'])
    nb = nbar_of(wb, T)
    jobs = [(N, g) for g in gs for N in Ns]
    R = _mapa_paralelo(_run2, jobs, procs, "Figura N2")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, (g, c) in enumerate(zip(gs, ['k', 'b', 'r', 'g'])):
        r = R[i*len(Ns):(i+1)*len(Ns)]
        S = (2*g/wb)**2; x = S*(2*nb+1)
        F = np.array([q['Fopt'] for q in r])
        eta = np.array([q['eta'] for q in r]); C = np.array([abs(q['C']) for q in r])
        ax[0].plot(Ns, 1-F, c, marker='o', ms=2.5, label=fr'$S={S:.3f}$')
        ax[0].plot(Ns, 2*Ns*x, c+'--', lw=1)
        ax[0].plot(Ns, 1-(1+(1-4*x)**Ns+(1-2*x)**Ns)/3, c+':', lw=1.2)
        ax[1].plot(Ns, eta, c, label=fr'$\eta$, $S={S:.3f}$'); ax[1].plot(Ns, C, c+'--', lw=1)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$N$'); ax[1].set_xlabel(r'$N$')
    ax[0].set_ylabel(r'$1-\bar F$')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.suptitle(fr'Ressonante; $\omega_b={wb:g}\gamma$, $k_BT={T:g}\hbar\gamma$ ($\bar n={nb:.3f}$), $\sigma={P["sig"]:g}\gamma$. '
                 r'Esq.: cheia = numérico, tracejada = $2NS(2\bar n+1)$, pontilhada = modelo exponencial', fontsize=9)
    plt.tight_layout(); plt.savefig(_out('figN2_vs_N.png'), dpi=150); plt.close(fig)


# ---------- Figura N3: mapa (omega_b, g) para N=10 ----------
N3 = dict(sig=0.1, T=1.0, N=10, wbs=np.geomspace(0.5, 30, 14), gs=np.geomspace(0.02, 1.5, 14))


def _run3(a):
    wb, g = a
    P = N3
    return cadeia_ressonante(P['sig'], P['N'], wb, g, nbar_of(wb, P['T']), L=81)['Fopt']


def fig3(procs):
    plt = _plt()
    P = N3; wbs, gs, T, N = P['wbs'], P['gs'], P['T'], P['N']
    jobs = [(wb, g) for g in gs for wb in wbs]
    F = np.array(_mapa_paralelo(_run3, jobs, procs, "Figura N3")).reshape(len(gs), len(wbs))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pc = ax.pcolormesh(wbs, gs, np.log10(np.clip(1-F, 1e-5, 1)), shading='gouraud', cmap='viridis')
    cs = ax.contour(wbs, gs, F, levels=[0.9, 0.99, 0.999], colors='w', linewidths=1)
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=7)
    W, G = np.meshgrid(wbs, gs); nb = 1/np.expm1(W/T)
    ax.contour(wbs, gs, N*(2*G/W)**2*(2*nb+1), levels=[0.01, 0.1], colors='r', linestyles='--', linewidths=1)
    ax.axvline(1.0, color='gray', lw=0.5, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\omega_b/\gamma$'); ax.set_ylabel(r'$g/\gamma$')
    plt.colorbar(pc, label=r'$\log_{10}(1-\bar F)$')
    ax.set_title(fr'$N={N}$ sítios, ressonante; $k_BT={T:g}\hbar\gamma$, $\sigma={P["sig"]:g}\gamma$. '
                 r'Vermelho: $NS(2\bar n+1)=0{,}01;\,0{,}1$', fontsize=8)
    plt.tight_layout(); plt.savefig(_out('figN3_mapa_N10.png'), dpi=150); plt.close(fig)


# ---------- Figura N4: largura do pacote ----------
N4 = dict(wb=10.0, g=0.5, T=5.0, Ns=[1, 5, 10], sigs=np.geomspace(0.005, 2, 18))


def _run4(a):
    sig, N = a
    P = N4
    r = cadeia_ressonante(sig, N, P['wb'], P['g'], nbar_of(P['wb'], P['T']))
    return (r['Fopt'], r['eta'], abs(r['C']))


def fig4(procs):
    plt = _plt()
    P = N4; sigs, Ns = P['sigs'], P['Ns']
    jobs = [(s, N) for N in Ns for s in sigs]
    R = np.array(_mapa_paralelo(_run4, jobs, procs, "Figura N4")).reshape(len(Ns), len(sigs), 3)
    nb = nbar_of(P['wb'], P['T']); S = (2*P['g']/P['wb'])**2
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, N in enumerate(Ns):
        c = CN[N]
        ax[0].plot(sigs, 1-R[i, :, 0], c, label=fr'$N={N}$')
        ax[0].axhline(2*N*S*(2*nb+1), color=c, ls='--', lw=0.8)
        ax[1].plot(sigs, R[i, :, 1], c, label=fr'$\eta$, $N={N}$'); ax[1].plot(sigs, R[i, :, 2], c+'--', lw=1)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$'); a.axvline(1, color='gray', lw=.5, ls=':')
        a.legend(fontsize=7)
    ax[0].set_yscale('log'); ax[0].set_ylabel(r'$1-\bar F$  (tracejada: $2NS(2\bar n+1)$)')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    fig.suptitle(fr'Ressonante; $\omega_b={P["wb"]:g}\gamma$, $g={P["g"]:g}\gamma$ ($S={S:.2f}$), '
                 fr'$k_BT={P["T"]:g}\hbar\gamma$', fontsize=10)
    plt.tight_layout(); plt.savefig(_out('figN4_vs_sigma.png'), dpi=150); plt.close(fig)


# ---------- Figura N5: temperatura ----------
N5 = dict(wb=10.0, g=0.3, sig=0.1, Ns=[1, 5, 10, 20], Ts=np.linspace(0.0, 30.0, 25))


def _run5(a):
    T, N = a
    P = N5
    return cadeia_ressonante(P['sig'], N, P['wb'], P['g'], nbar_of(P['wb'], T))['Fopt']


def fig5(procs):
    plt = _plt()
    P = N5; Ts, Ns = P['Ts'], P['Ns']
    jobs = [(T, N) for N in Ns for T in Ts]
    F = np.array(_mapa_paralelo(_run5, jobs, procs, "Figura N5")).reshape(len(Ns), len(Ts))
    S = (2*P['g']/P['wb'])**2
    nbs = np.array([nbar_of(P['wb'], T) for T in Ts])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, N in enumerate(Ns):
        c = CN[N]
        ax[0].plot(Ts, 1-F[i], c, marker='o', ms=2.5, label=fr'$N={N}$')
        ax[0].plot(Ts, 2*N*S*(2*nbs+1), c+'--', lw=1)
        ax[1].plot(nbs, 1-F[i], c, marker='o', ms=2.5, label=fr'$N={N}$')
        ax[1].plot(nbs, 2*N*S*(2*nbs+1), c+'--', lw=1)
    ax[0].set_xlabel(r'$k_BT/\hbar\gamma$'); ax[1].set_xlabel(r'$\bar n$')
    for a in ax:
        a.set_yscale('log'); a.set_ylabel(r'$1-\bar F$  (tracejada: $2NS(2\bar n+1)$)'); a.legend(fontsize=7)
    fig.suptitle(fr'Ressonante; $\omega_b={P["wb"]:g}\gamma$, $g={P["g"]:g}\gamma$ ($S={S:.4f}$), '
                 fr'$\sigma={P["sig"]:g}\gamma$', fontsize=10)
    plt.tight_layout(); plt.savefig(_out('figN5_vs_T.png'), dpi=150); plt.close(fig)


def _n_pontos(k):
    """Numero de pontos calculados em cada figura (para a barra global)."""
    if k == 1:
        nus = np.unique(np.concatenate([np.linspace(-13, -2, 120), np.linspace(-2, 2, 241),
                                        np.linspace(2, 13, 120)]))
        return len(N1['Ns']) * len(nus)
    return {2: len(N2['gs']) * len(N2['Ns']),
            3: len(N3['gs']) * len(N3['wbs']),
            4: len(N4['Ns']) * len(N4['sigs']),
            5: len(N5['Ns']) * len(N5['Ts'])}[k]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Figuras: 1 foton em N sitios com osciladores termicos.")
    ap.add_argument('--figs', type=int, nargs='+', default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--latex', choices=['auto', 'on', 'off'], default='auto',
                    help="auto: LaTeX real se houver latex+dvipng, senao mathtext Computer Modern")
    a = ap.parse_args()
    LATEX = a.latex
    print(f"diretorio de saida: {os.getcwd()}")
    print(f"texto: {'LaTeX (usetex)' if _usa_usetex() else 'mathtext Computer Modern (sem dvipng)'}")
    figs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}
    t_total = time.time()
    _iniciar_global(sum(_n_pontos(k) for k in a.figs))
    try:
        for i, k in enumerate(a.figs, 1):
            msg = f"[{i}/{len(a.figs)}] Figura N{k}: {_n_pontos(k)} pontos"
            (_G.barra.write if _G.barra is not None else print)(msg)
            t0 = time.time()
            figs[k](a.procs)
            (_G.barra.write if _G.barra is not None else print)(
                f"[{i}/{len(a.figs)}] Figura N{k} concluida em {_fmt_t(time.time() - t0)}")
    finally:
        _fechar_global()
    print(f"Total: {_fmt_t(time.time() - t_total)}")
