"""Figuras do regime lento (omega_b << gamma): 1 foton, N sitios, osciladores termicos.
Salva no diretorio corrente. Requer lento.py, nsitios.py e core.py na mesma pasta.

Uso:
    python lento_figs.py              # todas as figuras
    python lento_figs.py --figs 2 6   # so algumas
    python lento_figs.py --procs 4
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import progresso, _mapa_paralelo, _fmt_t, line_center  # noqa: E402
from lento import estatico, kappa, C1_fechado, nbar_of  # noqa: E402
from nsitios import cadeia  # noqa: E402

CN = {1: 'k', 2: 'b', 3: 'c', 5: 'r', 10: 'g', 20: 'm'}


def _out(nome):
    caminho = os.path.join(os.getcwd(), nome)
    print(f"salvo: {caminho}")
    return caminho


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------- Figura L1: varredura em frequencia (modelo quase-estatico) ----------
L1 = dict(wb=0.01, g=0.03, sig=0.1, T=0.02, Ns=[1, 2, 5, 10])


def fig1(procs):
    plt = _plt()
    P = L1; nb = nbar_of(P['wb'], P['T']); k = kappa(P['g'], nb); S = (2*P['g']/P['wb'])**2
    nus = np.linspace(-3, 3, 361)
    pts = [(N, nu) for N in P['Ns'] for nu in nus]
    R = [estatico(nu, P['sig'], N, P['g'], nb) for N, nu in progresso(pts, len(pts), "Figura L1")]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    for i, N in enumerate(P['Ns']):
        r = R[i*len(nus):(i+1)*len(nus)]; c = CN[N]; lab = fr'$N={N}$'
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
    ax[1, 0].axhline(2/3, color='gray', lw=0.6, ls=':')
    for a in ax.flat:
        a.axvline(0, color='gray', lw=0.5, ls=':'); a.set_xlabel(r'$\omega_c/\gamma$'); a.legend(fontsize=7)
    ax[0, 0].set_ylabel(r'$\eta$ (população no modo ideal)')
    ax[0, 1].set_ylabel(r'$|C|$ (coerência)')
    ax[1, 0].set_ylabel(r'$\bar F$ (cheia) e com correção de fase (tracejada)')
    ax[1, 1].set_ylabel(r'erro de fase $\arg C/\pi$')
    fig.suptitle(fr'Regime lento; $\omega_b={P["wb"]:g}\gamma$, $g={P["g"]:g}\gamma$ ($S={S:.0f}$), '
                 fr'$k_BT={P["T"]:g}\hbar\gamma$ ($\bar n={nb:.2f}$), $\kappa={k:.3f}$, $\sigma={P["sig"]:g}\gamma$',
                 fontsize=10)
    plt.tight_layout(); plt.savefig(_out('figL1_vs_freq.png'), dpi=150); plt.close(fig)


# ---------- Figura L2: acumulo com N ----------
L2 = dict(wb=0.01, sig=0.1, T=0.02, gs=[0.005, 0.01, 0.02, 0.04], Ns=list(range(1, 61)))


def _run2(a):
    N, g = a
    P = L2
    return estatico(0.0, P['sig'], N, g, nbar_of(P['wb'], P['T']))


def fig2(procs):
    plt = _plt()
    P = L2; nb = nbar_of(P['wb'], P['T']); Ns = np.array(P['Ns'])
    jobs = [(N, g) for g in P['gs'] for N in Ns]
    R = _mapa_paralelo(_run2, jobs, procs, "Figura L2")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, (g, c) in enumerate(zip(P['gs'], ['k', 'b', 'r', 'g'])):
        r = R[i*len(Ns):(i+1)*len(Ns)]; k = kappa(g, nb)
        F = np.array([q['Fopt'] for q in r]); eta = np.array([q['eta'] for q in r])
        C = np.array([abs(q['C']) for q in r])
        ax[0].plot(Ns, 1-F, c, marker='o', ms=2.0, label=fr'$\kappa={k:.4f}$')
        ax[0].plot(Ns, (2/3)*Ns*k, c+'--', lw=1)
        ax[0].plot(Ns, 1-(2+C1_fechado(k)**Ns)/3, c+':', lw=1.4)
        ax[1].plot(Ns, eta, c, label=fr'$\eta$, $\kappa={k:.4f}$'); ax[1].plot(Ns, C, c+'--', lw=1)
    ax[0].axhline(1/3, color='gray', lw=0.6)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    for a in ax:
        a.set_xlabel(r'$N$'); a.legend(fontsize=7)
    ax[0].set_ylabel(r'$1-\bar F$')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    fig.suptitle(fr'Regime lento, na linha; $\omega_b={P["wb"]:g}\gamma$, $k_BT={P["T"]:g}\hbar\gamma$ '
                 fr'($\bar n={nb:.2f}$), $\sigma={P["sig"]:g}\gamma$. Esq.: tracejada $=\frac{{2}}{{3}}N\kappa$, '
                 r'pontilhada $=1-(2+C_1^N)/3$, cinza $=1/3$', fontsize=9)
    plt.tight_layout(); plt.savefig(_out('figL2_vs_N.png'), dpi=150); plt.close(fig)


# ---------- Figura L3: mapa (omega_b, g) para N=10 ----------
L3 = dict(sig=0.1, T=0.02, N=10, wbs=np.geomspace(1e-3, 0.03, 24), gs=np.geomspace(1e-3, 0.1, 24))


def _run3(a):
    wb, g = a
    P = L3
    return estatico(0.0, P['sig'], P['N'], g, nbar_of(wb, P['T']), L=101, M=161)['Fopt']


def fig3(procs):
    plt = _plt()
    P = L3; wbs, gs, T, N = P['wbs'], P['gs'], P['T'], P['N']
    jobs = [(wb, g) for g in gs for wb in wbs]
    F = np.array(_mapa_paralelo(_run3, jobs, procs, "Figura L3")).reshape(len(gs), len(wbs))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pc = ax.pcolormesh(wbs, gs, np.log10(np.clip(1-F, 1e-6, 1)), shading='gouraud', cmap='viridis')
    cs = ax.contour(wbs, gs, F, levels=[0.9, 0.99, 0.999], colors='w', linewidths=1)
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=7)
    W, G = np.meshgrid(wbs, gs); nb = 1/np.expm1(W/T)
    ax.contour(wbs, gs, N*(4*G)**2*(2*nb+1), levels=[0.015, 0.15], colors='r', linestyles='--', linewidths=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'$\omega_b/\gamma$'); ax.set_ylabel(r'$g/\gamma$')
    plt.colorbar(pc, label=r'$\log_{10}(1-\bar F)$')
    ax.set_title(fr'Regime lento, $N={N}$; $k_BT={T:g}\hbar\gamma$, $\sigma={P["sig"]:g}\gamma$. '
                 r'Vermelho: $\frac{2}{3}N\kappa=0{,}01;\,0{,}1$', fontsize=8)
    plt.tight_layout(); plt.savefig(_out('figL3_mapa_N10.png'), dpi=150); plt.close(fig)


# ---------- Figura L4: largura do pacote (calculo EXATO; mostra sigma ~ omega_b) ----------
L4 = dict(wb=0.05, g=0.04, T=0.0, Ns=[1, 5, 10], sigs=np.geomspace(0.003, 1.0, 20))


def _run4(a):
    sig, N = a
    P = L4; nb = nbar_of(P['wb'], P['T'])
    c = line_center(P['wb'], P['g'], nb)
    r = cadeia(c, sig, N, P['wb'], P['g'], nb, Dref=c, L=121)
    return (r['Fopt'], r['eta'], abs(r['C']))


def fig4(procs):
    plt = _plt()
    P = L4; sigs, Ns = P['sigs'], P['Ns']; nb = nbar_of(P['wb'], P['T']); k = kappa(P['g'], nb)
    jobs = [(s, N) for N in Ns for s in sigs]
    R = np.array(_mapa_paralelo(_run4, jobs, procs, "Figura L4")).reshape(len(Ns), len(sigs), 3)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, N in enumerate(Ns):
        c = CN[N]
        ax[0].plot(sigs, 1-R[i, :, 0], c, marker='o', ms=2.5, label=fr'$N={N}$')
        ax[0].axhline(1-(2+C1_fechado(k)**N)/3, color=c, ls='--', lw=0.8)
        CN_ = C1_fechado(k)**N
        ax[0].axhline(1-(1+CN_**2+CN_)/3, color=c, ls=':', lw=0.8)
        ax[1].plot(sigs, R[i, :, 1], c, label=fr'$\eta$, $N={N}$'); ax[1].plot(sigs, R[i, :, 2], c+'--', lw=1)
    for a in ax:
        a.set_xscale('log'); a.set_xlabel(r'$\sigma/\gamma$')
        a.axvline(P['wb'], color='gray', lw=0.8, ls='-'); a.axvline(1, color='gray', lw=.5, ls=':')
        a.legend(fontsize=7)
    ax[0].set_yscale('log'); ax[0].set_ylabel(r'$1-\bar F$')
    ax[1].set_ylabel(r'$\eta$ (cheia), $|C|$ (tracejada)')
    fig.suptitle(fr'Cálculo exato; $\omega_b={P["wb"]:g}\gamma$, $g={P["g"]:g}\gamma$, $k_BT=0$ ($\kappa={k:.4f}$). '
                 r'Linha cinza: $\sigma=\omega_b$. Esq.: tracejada $=$ previsão $\sigma\gg\omega_b$, '
                 r'pontilhada $=$ previsão $\sigma\ll\omega_b$', fontsize=9)
    plt.tight_layout(); plt.savefig(_out('figL4_vs_sigma.png'), dpi=150); plt.close(fig)


# ---------- Figura L5: temperatura ----------
L5 = dict(wb=0.01, g=0.01, sig=0.1, Ns=[1, 5, 10, 20], Ts=np.linspace(0.0, 0.2, 26))


def _run5(a):
    T, N = a
    P = L5
    return estatico(0.0, P['sig'], N, P['g'], nbar_of(P['wb'], T))['Fopt']


def fig5(procs):
    plt = _plt()
    P = L5; Ts, Ns = P['Ts'], P['Ns']
    jobs = [(T, N) for N in Ns for T in Ts]
    F = np.array(_mapa_paralelo(_run5, jobs, procs, "Figura L5")).reshape(len(Ns), len(Ts))
    nbs = np.array([nbar_of(P['wb'], T) for T in Ts]); ks = kappa(P['g'], nbs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for i, N in enumerate(Ns):
        c = CN[N]
        ax[0].plot(Ts, 1-F[i], c, marker='o', ms=2.5, label=fr'$N={N}$')
        ax[0].plot(Ts, (2/3)*N*ks, c+'--', lw=1)
        ax[1].plot(nbs, 1-F[i], c, marker='o', ms=2.5, label=fr'$N={N}$')
        ax[1].plot(nbs, (2/3)*N*ks, c+'--', lw=1)
    ax[0].set_xlabel(r'$k_BT/\hbar\gamma$'); ax[1].set_xlabel(r'$\bar n$')
    for a in ax:
        a.set_yscale('log'); a.set_ylabel(r'$1-\bar F$  (tracejada: $\frac{2}{3}N\kappa$)'); a.legend(fontsize=7)
    fig.suptitle(fr'Regime lento, na linha; $\omega_b={P["wb"]:g}\gamma$, $g={P["g"]:g}\gamma$, '
                 fr'$\sigma={P["sig"]:g}\gamma$', fontsize=10)
    plt.tight_layout(); plt.savefig(_out('figL5_vs_T.png'), dpi=150); plt.close(fig)


# ---------- Figura L6: a relacao unificada (exato x formula) e a validacao do modelo estatico ----------
L6 = dict(g=0.01, wbs=np.geomspace(0.01, 30, 25), nbs=[0.0, 0.5],
          val=dict(wb=0.05, g=0.04, Ns=list(range(1, 11)), sigs=[0.25, 0.01]))


def _run6a(a):
    wb, nb = a
    g = L6['g']
    c = line_center(wb, g, nb)
    r = cadeia(c, 0.002, 1, wb, g, nb, Dref=c, L=121)
    return 1 - r['C'].real


def _run6b(a):
    N, sig = a
    V = L6['val']
    c = line_center(V['wb'], V['g'], 0.0)
    ex = cadeia(c, sig, N, V['wb'], V['g'], 0.0, Dref=c, L=121)
    es = estatico(0.0, sig, N, V['g'], 0.0)
    Cs = es['C'].real
    return (ex['Fbar'], es['Fbar'], (1 + Cs**2 + Cs)/3, abs(ex['C']))


def fig6(procs):
    plt = _plt()
    P = L6; wbs = P['wbs']; g = P['g']
    jobs = [(wb, nb) for nb in P['nbs'] for wb in wbs]
    E = np.array(_mapa_paralelo(_run6a, jobs, procs, "Figura L6a")).reshape(len(P['nbs']), len(wbs))
    V = P['val']
    jobs = [(N, s) for s in V['sigs'] for N in V['Ns']]
    Q = np.array(_mapa_paralelo(_run6b, jobs, procs, "Figura L6b")).reshape(len(V['sigs']), len(V['Ns']), 4)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    wf = np.geomspace(wbs[0], wbs[-1], 400)
    for i, (nb, c) in enumerate(zip(P['nbs'], ['k', 'r'])):
        ax[0].plot(wbs, E[i], c+'o', ms=4, label=fr'exato, $\bar n={nb:g}$')
        ax[0].plot(wf, 32*g**2*(2*nb+1)/(1+4*wf**2), c, lw=1.2, label=r'$32g^2(2\bar n+1)/(\gamma^2+4\omega_b^2)$' if i == 0 else None)
        ax[0].plot(wf, 2*(2*g/wf)**2*(2*nb+1), c+'--', lw=0.8, label=r'rápido: $2S(2\bar n+1)$' if i == 0 else None)
        ax[0].axhline(32*g**2*(2*nb+1), color=c, ls=':', lw=0.8)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(3e-6, 3e-2)
    ax[0].axvline(0.5, color='gray', lw=0.5, ls=':')
    ax[0].set_xlabel(r'$\omega_b/\gamma$'); ax[0].set_ylabel(r'perda de coerência por sítio $1-\mathrm{Re}\,C_1$')
    ax[0].set_title(fr'$g={g:g}\gamma$; pontilhada: lento $2\kappa$', fontsize=9)
    ax[0].legend(fontsize=7)
    Ns = np.array(V['Ns'])
    for j, (s, c) in enumerate(zip(V['sigs'], ['b', 'g'])):
        ax[1].plot(Ns, 1-Q[j, :, 0], c+'o', ms=4, label=fr'exato, $\sigma={s:g}\gamma$')
        ax[1].plot(Ns, 1-Q[j, :, 1], c, lw=1, label=fr'estático $(2+C)/3$, $\sigma={s:g}\gamma$')
        ax[1].plot(Ns, 1-Q[j, :, 2], c+'--', lw=1, label=fr'$(1+C^2+C)/3$, $\sigma={s:g}\gamma$')
    ax[1].set_xlabel(r'$N$'); ax[1].set_ylabel(r'$1-\bar F$')
    ax[1].set_title(fr'$\omega_b={V["wb"]:g}\gamma$, $g={V["g"]:g}\gamma$, $k_BT=0$', fontsize=9)
    ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(_out('figL6_relacao_validacao.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Figuras do regime lento.")
    ap.add_argument('--figs', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    print(f"diretorio de saida: {os.getcwd()}")
    t0 = time.time()
    for k in a.figs:
        {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}[k](a.procs)
    print(f"Total: {_fmt_t(time.time() - t0)}")
