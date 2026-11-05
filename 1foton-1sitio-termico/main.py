"""Qubit dual-rail de 1 foton, um sitio de Brod-Combes com um oscilador (banho) termico.

Uso:
    python core.py              # gera as 4 figuras no diretorio corrente
    python core.py --figs 1 3   # so as figuras 1 e 3
    python core.py --procs 4    # numero de processos para as varreduras (figs 2-4)
As imagens sao salvas no diretorio de onde o comando e executado (os.getcwd()).

Trilho 1 passa pelo atomo (acoplado ao oscilador b); trilho 2 e livre.
Ideal: trilho 1 com a matriz-S de um foton SEM banho (atomo na linha deslocada
pelo polaron, calibravel), trilho 2 identidade.
"""
import os
import sys
import time
import argparse
import numpy as np
from scipy.linalg import expm
from multiprocessing import Pool

gam = 1.0


def disp_matrix(beta, N):
    b = np.diag(np.sqrt(np.arange(1, N)), 1)
    return expm(beta * (b.T - b))  # D[k, n] = <k|D(beta)|n>


def thermal_p(nbar, nmax):
    n = np.arange(nmax + 1)
    if nbar <= 0:
        p = np.zeros(nmax + 1); p[0] = 1.0
        return p
    return nbar ** n / (nbar + 1) ** (n + 1)


def t_mn(m, n, nu, Delta, wb, g, D, K):
    """Elemento da Eq. (21) do PDF de Fock, vetorizado em nu."""
    ks = np.arange(K)
    num = D[ks, m] * D[ks, n]
    den = gam / 2 + 1j * (Delta - 4 * g**2 / wb + (ks[:, None] - n) * wb - nu[None, :])
    return (1.0 if m == n else 0.0) - gam * (num[:, None] / den).sum(0)


def t0(nu, Delta_eff):
    """Sem banho, atomo em Delta_eff: Eq. (24) de Brod-Combes."""
    G = gam / 2 + 1j * (Delta_eff - nu)
    return -np.conj(G) / G


def xi(w, wc, sig):
    return (2 * np.pi * sig**2) ** -0.25 * np.exp(-(w - wc) ** 2 / (4 * sig**2))


def quantities(wc, sig, Delta, wb, g, nbar, polaron_ref=True, nmax=None, side=None, Ngrid=4001, Dref=None):
    """Retorna eta (populacao no modo ideal), C (coerencia), Fbar, Fbar_fase, fase ideal,
    e o peso total de bandas laterais.
    a_mn = <u1|phi_mn>,  u1 = xi*t0,  phi_mn(w) = xi(w+(m-n)wb) t_mn(w+(m-n)wb)."""
    if nmax is None:
        nmax = int(max(6, nbar * 12 + 6))
    if side is None:
        side = int(min(40, 8 + 6 * (2 * g / wb) ** 2 * (2 * nbar + 1) + 4))
    p = thermal_p(nbar, nmax)
    p = p / p.sum()
    beta = 2 * g / wb
    N = nmax + side + 60
    D = disp_matrix(beta, N)
    K = N - 20
    if Dref is None:
        Dref = Delta - 4 * g**2 / wb if polaron_ref else Delta
    W = 8 * sig + 2 * gam
    w = np.linspace(wc - W, wc + W, Ngrid); dw = w[1] - w[0]
    u1 = xi(w, wc, sig) * t0(w, Dref)
    eta = 0.0; C = 0.0 + 0j; side_w = 0.0; unit = 0.0
    for n in range(nmax + 1):
        if p[n] < 1e-12:
            continue
        for m in range(max(0, n - side), n + side + 1):
            sh = (m - n) * wb
            nu = w + sh
            phi = xi(nu, wc, sig) * t_mn(m, n, nu, Delta, wb, g, D, K)
            a = np.sum(np.conj(u1) * phi) * dw
            eta += p[n] * abs(a) ** 2
            wt = np.sum(abs(xi(w, wc, sig) * t_mn(m, n, w, Delta, wb, g, D, K)) ** 2) * dw
            unit += p[n] * wt
            if m == n:
                C += p[n] * a
            else:
                side_w += p[n] * wt
    Fbar = (1 + eta + C.real) / 3
    Fopt = (1 + eta + abs(C)) / 3
    phase_id = np.angle(np.sum(abs(xi(w, wc, sig)) ** 2 * t0(w, Dref)) * dw)
    return dict(eta=eta, C=C, Fbar=Fbar, Fopt=Fopt, phase=phase_id, side=side_w, unit=unit)


def nbar_of(wb, T):
    return 0.0 if T == 0 else 1.0 / np.expm1(wb / T)


def line_center(wb, g, nbar, Delta=0.0):
    """Frequencia em que a resposta coerente media sum_n p_n t_nn(nu) tem fase pi
    (o foton ganha a fase de ressonancia). Escolhe a raiz de interacao mais forte."""
    nmax = int(max(6, nbar * 12 + 6)); p = thermal_p(nbar, nmax); p /= p.sum()
    N = nmax + 100; D = disp_matrix(2 * g / wb, N); K = N - 20
    nu = np.linspace(Delta - 4 * g**2 / wb - 3, Delta + 3, 6001)
    tc = sum(p[n] * t_mn(n, n, nu, Delta, wb, g, D, K) for n in range(nmax + 1) if p[n] > 1e-12)
    im = tc.imag; idx = np.where((np.sign(im[:-1]) != np.sign(im[1:])) & (tc.real[:-1] < 0))[0]
    if len(idx) == 0:
        return nu[np.argmin(tc.real)]
    best = idx[np.argmin(tc.real[idx])]
    x0, x1 = nu[best], nu[best + 1]; y0, y1 = im[best], im[best + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def resonant(sig, wb, g, nbar, Ngrid=801):
    c = line_center(wb, g, nbar)
    r = quantities(c, sig, 0.0, wb, g, nbar, Ngrid=Ngrid, Dref=c); r['wc'] = c
    return r


# =====================================================================
#  Barra de progresso com estimativa de tempo
#  (usa tqdm se estiver instalado; senao, uma barra simples sem dependencias)
# =====================================================================

def _fmt_t(s):
    s = int(round(s))
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


class _Barra:
    def __init__(self, total, desc, largura=30):
        self.total, self.desc, self.largura = total, desc, largura
        self.n, self.t0 = 0, time.time()
        self._mostrar()

    def update(self, k=1):
        self.n += k
        self._mostrar()

    def _mostrar(self):
        frac = self.n / self.total if self.total else 1.0
        cheio = int(self.largura * frac)
        dt = time.time() - self.t0
        if self.n > 0:
            resta = dt * (self.total - self.n) / self.n
            eta = f"restam ~{_fmt_t(resta)}"
        else:
            eta = "estimando..."
        barra = "#" * cheio + "-" * (self.largura - cheio)
        sys.stdout.write(f"\r{self.desc}: [{barra}] {self.n}/{self.total} "
                         f"({100*frac:5.1f}%)  decorrido {_fmt_t(dt)}  {eta}   ")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write(f"\n{self.desc}: concluido em {_fmt_t(time.time() - self.t0)}\n")
        sys.stdout.flush()


def progresso(iteravel, total, desc):
    """Itera mostrando barra de progresso, % concluido, tempo decorrido e tempo restante."""
    try:
        from tqdm import tqdm
        yield from tqdm(iteravel, total=total, desc=desc,
                        bar_format="{desc}: {percentage:5.1f}%|{bar:30}| {n}/{total} "
                                   "[decorrido {elapsed}, restam ~{remaining}]")
        return
    except ImportError:
        pass
    b = _Barra(total, desc)
    for x in iteravel:
        yield x
        b.update()
    b.close()


def _mapa_paralelo(func, jobs, procs, desc):
    """Pool.imap (mantem a ordem) com barra de progresso."""
    with Pool(procs) as P:
        return list(progresso(P.imap(func, jobs, chunksize=1), len(jobs), desc))


# =====================================================================
#  Figuras (salvas no diretorio corrente)
# =====================================================================

def _out(nome):
    caminho = os.path.join(os.getcwd(), nome)
    print(f"salvo: {caminho}")
    return caminho


# Estilo igual ao de 2foton-1sitio-fock/plot_fock.py
CORES = ['black', 'blue', 'red', '#8A2BE2']
LW = 1.2


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.0,
        'svg.fonttype': 'path',
        'lines.linewidth': LW,
    })
    return plt


def _salvar(plt, fig, nome):
    """Salva em .svg e .png (300 dpi), como em plot_fock.py."""
    fig.tight_layout()
    base = os.path.splitext(nome)[0]
    fig.savefig(_out(base + '.svg'), dpi=300, bbox_inches='tight')
    fig.savefig(_out(base + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------- Figura 1: varredura em frequencia ----------
def fig1():
    plt = _plt()
    Delta, wb, g, sig = 0.0, 3.0, 0.75, 0.1
    S = (2*g/wb)**2
    nus = np.linspace(-5, 5, 401)
    # fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))   # versao com populacao e coerencia
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    temps = [(0.0, CORES[0]), (1.0, CORES[1]), (5.0, CORES[2])]
    pontos = [(T, nu) for T, _ in temps for nu in nus]
    todos = [quantities(nu, sig, Delta, wb, g, nbar_of(wb, T), Ngrid=601)
             for T, nu in progresso(pontos, len(pontos), "Figura 1")]
    for i, (T, c) in enumerate(temps):
        nb = nbar_of(wb, T)
        R = todos[i*len(nus):(i+1)*len(nus)]
        # eta = np.array([r['eta'] for r in R]); C = np.array([r['C'] for r in R])
        F = np.array([r['Fbar'] for r in R]); Fo = np.array([r['Fopt'] for r in R])
        lab = fr'$k_BT/\hbar\gamma={T:g}$ ($\bar n={nb:.2f}$)'
        # ax[0, 0].plot(nus, eta, '-', color=c, label=lab)
        # ax[0, 1].plot(nus, abs(C), '-', color=c, label=lab)
        ax[0].plot(nus, F, '-', color=c, label=lab); ax[0].plot(nus, Fo, '--', color=c)
        if T == 0:
            ph = np.array([r['phase'] for r in R])
            ax[1].plot(nus, np.mod(ph, 2*np.pi)/np.pi, '-', color=CORES[0])
    for a in ax.flat:
        for k in range(-2, 3):
            a.axvline(Delta - 4*g**2/wb + k*wb, color='gray', lw=0.5, ls=':')
        a.set_xlabel(r'$\omega_c/\gamma$')
    # ax[0, 0].set_ylabel(r'$\eta=\rho_{11}$ (população no modo ideal)')
    # ax[0, 1].set_ylabel(r'$|C|=|\rho_{12}|/|\alpha\beta|$ (coerência)')
    ax[0].set_ylabel(r'$\bar F$ (cheia) e $\bar F$ com correção de fase (tracejada)')
    ax[1].set_ylabel(r'fase ideal $\arg t_0$ $/\pi$')
    ax[0].legend(fontsize=8)
    fig.suptitle(fr'$\omega_b={wb:g}\gamma$, $g={g:g}\gamma$, $S={S:.2f}$, $\sigma={sig:g}\gamma$', fontsize=10)
    _salvar(plt, fig, 'fig1_vs_freq')


# ---------- Figura 2: frequencia do oscilador ----------
F2_SIG, F2_G = 0.1, 0.5
F2_WBS = np.geomspace(0.5, 30, 22)
F2_NBS = [0.0, 0.5, 2.0]


def _run2(args):
    wb, nb = args
    return resonant(F2_SIG, wb, F2_G, nb, Ngrid=601)


def fig2(procs):
    plt = _plt()
    sig, g, wbs, nbs = F2_SIG, F2_G, F2_WBS, F2_NBS
    jobs = [(wb, nb) for nb in nbs for wb in wbs]
    R = _mapa_paralelo(_run2, jobs, procs, "Figura 2")
    R = np.array(R).reshape(len(nbs), len(wbs))
    # fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))   # versao com populacao e coerencia
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax = [ax]
    for i, (nb, c) in enumerate(zip(nbs, CORES)):
        F = np.array([r['Fopt'] for r in R[i]])
        # eta = np.array([r['eta'] for r in R[i]])
        # C = np.array([abs(r['C']) for r in R[i]])
        S = (2*g/wbs)**2
        ax[0].plot(wbs, 1-F, '-', color=c, label=fr'$\bar n={nb:g}$')
        ax[0].plot(wbs, 2*S*(2*nb+1), '--', color=c)
        # ax[1].plot(wbs, eta, '-', color=c, label=fr'$\eta$, $\bar n={nb:g}$')
        # ax[1].plot(wbs, C, '--', color=c, label=fr'$|C|$, $\bar n={nb:g}$')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')  # ; ax[1].set_xscale('log')
    for a in ax:
        a.axvline(1.0, color='gray', lw=0.5, ls=':'); a.set_xlabel(r'$\omega_b/\gamma$')
    ax[0].set_ylabel(r'$1-\bar F$')
    # ax[1].set_ylabel(r'população $\eta$ (cheia), coerência $|C|$ (tracejada)')
    ax[0].legend(fontsize=8)  # ; ax[1].legend(fontsize=8)
    fig.suptitle(fr'Ressonante (fase coerente $=\pi$); $g={g:g}\gamma$, $\sigma={sig:g}\gamma$', fontsize=10)
    _salvar(plt, fig, 'fig2_vs_wb')


# ---------- Figura 3: mapa (omega_b, g) em T fixo ----------
F3_SIG, F3_T = 0.1, 1.0
F3_WBS = np.geomspace(0.5, 30, 14)
F3_GS = np.geomspace(0.05, 1.5, 14)


def _run3(args):
    wb, g = args
    return resonant(F3_SIG, wb, g, nbar_of(wb, F3_T), Ngrid=401)['Fopt']


def fig3(procs):
    plt = _plt()
    sig, T, wbs, gs = F3_SIG, F3_T, F3_WBS, F3_GS
    jobs = [(wb, g) for g in gs for wb in wbs]
    F = np.array(_mapa_paralelo(_run3, jobs, procs, "Figura 3")).reshape(len(gs), len(wbs))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pc = ax.pcolormesh(wbs, gs, np.log10(np.clip(1-F, 1e-5, 1)), shading='gouraud', cmap='viridis')
    cs = ax.contour(wbs, gs, F, levels=[0.9, 0.99, 0.999], colors='w', linewidths=LW)
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=8)
    W, G = np.meshgrid(wbs, gs); nb = 1/np.expm1(W/T)
    ax.contour(wbs, gs, (2*G/W)**2*(2*nb+1), levels=[0.01, 0.1], colors='red', linestyles='--', linewidths=LW)
    ax.axvline(1.0, color='gray', lw=0.5, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\omega_b/\gamma$'); ax.set_ylabel(r'$g/\gamma$')
    plt.colorbar(pc, label=r'$\log_{10}(1-\bar F)$')
    ax.set_title(fr'Ressonante (fase coerente $=\pi$); $k_BT={T:g}\hbar\gamma$, $\sigma={sig:g}\gamma$. '
                 r'Vermelho: $S(2\bar n+1)=0{,}01;\,0{,}1$', fontsize=10)
    _salvar(plt, fig, 'fig3_mapa')


# ---------- Figura 4: largura do pacote ----------
F4_T = 1.0
F4_SIGS = np.geomspace(0.01, 3, 18)
F4_CASES = [(10.0, 0.5, CORES[0]), (3.0, 0.3, CORES[1]), (0.5, 0.1, CORES[2])]


def _run4(a):
    sig, wb, g = a
    r = resonant(sig, wb, g, nbar_of(wb, F4_T), Ngrid=801)
    return r['Fopt']
    # return (r['Fopt'], r['eta'], abs(r['C']))   # versao com populacao e coerencia


def fig4(procs):
    plt = _plt()
    T, sigs, cases = F4_T, F4_SIGS, F4_CASES
    jobs = [(s, wb, g) for wb, g, _ in cases for s in sigs]
    R = np.array(_mapa_paralelo(_run4, jobs, procs, "Figura 4")).reshape(len(cases), len(sigs))
    # fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))   # versao com populacao e coerencia
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax = [ax]
    for i, (wb, g, c) in enumerate(cases):
        nb = nbar_of(wb, T); S = (2*g/wb)**2
        lab = fr'$\omega_b={wb:g}\gamma$, $S={S:.2f}$, $\bar n={nb:.2f}$'
        ax[0].plot(sigs, 1-R[i], '-', color=c, label=lab)
        # ax[1].plot(sigs, R[i, :, 1], '-', color=c, label=r'$\eta$, '+lab)
        # ax[1].plot(sigs, R[i, :, 2], '--', color=c)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$'); a.axvline(1, color='gray', lw=.5, ls=':')
    ax[0].set_yscale('log'); ax[0].set_ylabel(r'$1-\bar F$')
    # ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    ax[0].legend(fontsize=8)  # ; ax[1].legend(fontsize=7)
    fig.suptitle(fr'Ressonante (fase coerente $=\pi$); $k_BT={T:g}\hbar\gamma$', fontsize=10)
    _salvar(plt, fig, 'fig4_vs_sigma')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Gera as figuras do qubit dual-rail com banho no diretorio corrente.")
    ap.add_argument('--figs', type=int, nargs='+', default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    print(f"diretorio de saida: {os.getcwd()}")
    t_total = time.time()
    for k in a.figs:
        if k == 1:
            fig1()
        else:
            {2: fig2, 3: fig3, 4: fig4}[k](a.procs)
    print(f"Total: {_fmt_t(time.time() - t_total)}")