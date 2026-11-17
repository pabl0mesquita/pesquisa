"""Calculo das figuras do regime lento (omega_b << gamma): 1 foton, N sitios, osciladores termicos.
Salva os resultados em ./data/figL{k}.npz (diretorio corrente) para plotar_lento.py.
Requer lento.py, nsitios.py e core.py na mesma pasta.

Uso:
    python calcular_lento.py              # todas as figuras
    python calcular_lento.py --figs 2 6   # so algumas
    python calcular_lento.py --procs 4
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import progresso, _mapa_paralelo, _fmt_t, line_center  # noqa: E402
from lento import estatico, kappa, nbar_of  # noqa: E402
from nsitios import cadeia  # noqa: E402

PASTA = 'data'   # relativa ao diretorio corrente


def _salvar(nome, **dados):
    os.makedirs(PASTA, exist_ok=True)
    caminho = os.path.join(os.getcwd(), PASTA, nome)
    np.savez(caminho, **dados)
    print(f"salvo: {caminho}")
    return caminho


# =====================================================================
#  Parametros de cada figura
# =====================================================================
L1 = dict(wb=0.01, g=0.03, sig=0.1, T=0.02, Ns=[1, 2, 5, 10], nus=np.linspace(-3, 3, 361))
L2 = dict(wb=0.01, sig=0.1, T=0.02, gs=[0.005, 0.01, 0.02, 0.04], Ns=list(range(1, 61)))
L3 = dict(sig=0.1, T=0.02, N=10, wbs=np.geomspace(1e-3, 0.03, 24), gs=np.geomspace(1e-3, 0.1, 24))
L4 = dict(wb=0.05, g=0.04, T=0.0, Ns=[1, 5, 10], sigs=np.geomspace(0.003, 1.0, 20))
L5 = dict(wb=0.01, g=0.01, sig=0.1, Ns=[1, 5, 10, 20], Ts=np.linspace(0.0, 0.2, 26))
L6 = dict(g=0.01, wbs=np.geomspace(0.01, 30, 25), nbs=[0.0, 0.5],
          val=dict(wb=0.05, g=0.04, Ns=list(range(1, 11)), sigs=[0.25, 0.01]))


# ---------- Figura L1: varredura em frequencia (modelo quase-estatico) ----------
def calc1(procs):
    P = L1; nb = nbar_of(P['wb'], P['T']); nus = P['nus']; Ns = P['Ns']
    pts = [(N, nu) for N in Ns for nu in nus]
    R = [estatico(nu, P['sig'], N, P['g'], nb) for N, nu in progresso(pts, len(pts), "Figura L1")]

    def campo(k):
        return np.array([q[k] for q in R]).reshape(len(Ns), len(nus))

    _salvar('figL1.npz', nus=nus, Ns=np.array(Ns),
            eta=campo('eta'), C=campo('C'), Fbar=campo('Fbar'), Fopt=campo('Fopt'), phase=campo('phase'),
            wb=P['wb'], g=P['g'], sig=P['sig'], T=P['T'], nb=nb,
            kappa=kappa(P['g'], nb), S=(2*P['g']/P['wb'])**2)


# ---------- Figura L2: acumulo com N ----------
def _run2(a):
    N, g = a
    P = L2
    return estatico(0.0, P['sig'], N, g, nbar_of(P['wb'], P['T']))


def calc2(procs):
    P = L2; nb = nbar_of(P['wb'], P['T']); Ns = np.array(P['Ns']); gs = P['gs']
    jobs = [(N, g) for g in gs for N in Ns]
    R = _mapa_paralelo(_run2, jobs, procs, "Figura L2")
    sh = (len(gs), len(Ns))
    _salvar('figL2.npz', Ns=Ns, gs=np.array(gs),
            Fopt=np.array([q['Fopt'] for q in R]).reshape(sh),
            eta=np.array([q['eta'] for q in R]).reshape(sh),
            absC=np.array([abs(q['C']) for q in R]).reshape(sh),
            kappas=np.array([kappa(g, nb) for g in gs]),
            wb=P['wb'], sig=P['sig'], T=P['T'], nb=nb)


# ---------- Figura L3: mapa (omega_b, g) para N=10 ----------
def _run3(a):
    wb, g = a
    P = L3
    return estatico(0.0, P['sig'], P['N'], g, nbar_of(wb, P['T']), L=101, M=161)['Fopt']


def calc3(procs):
    P = L3; wbs, gs = P['wbs'], P['gs']
    jobs = [(wb, g) for g in gs for wb in wbs]
    F = np.array(_mapa_paralelo(_run3, jobs, procs, "Figura L3")).reshape(len(gs), len(wbs))
    _salvar('figL3.npz', wbs=wbs, gs=gs, F=F, sig=P['sig'], T=P['T'], N=P['N'])


# ---------- Figura L4: largura do pacote (calculo EXATO; mostra sigma ~ omega_b) ----------
def _run4(a):
    sig, N = a
    P = L4; nb = nbar_of(P['wb'], P['T'])
    c = line_center(P['wb'], P['g'], nb)
    r = cadeia(c, sig, N, P['wb'], P['g'], nb, Dref=c, L=121)
    return (r['Fopt'], r['eta'], abs(r['C']))


def calc4(procs):
    P = L4; sigs, Ns = P['sigs'], P['Ns']; nb = nbar_of(P['wb'], P['T'])
    jobs = [(s, N) for N in Ns for s in sigs]
    R = np.array(_mapa_paralelo(_run4, jobs, procs, "Figura L4")).reshape(len(Ns), len(sigs), 3)
    _salvar('figL4.npz', sigs=sigs, Ns=np.array(Ns),
            Fopt=R[:, :, 0], eta=R[:, :, 1], absC=R[:, :, 2],
            wb=P['wb'], g=P['g'], T=P['T'], nb=nb, kappa=kappa(P['g'], nb))


# ---------- Figura L5: temperatura ----------
def _run5(a):
    T, N = a
    P = L5
    return estatico(0.0, P['sig'], N, P['g'], nbar_of(P['wb'], T))['Fopt']


def calc5(procs):
    P = L5; Ts, Ns = P['Ts'], P['Ns']
    jobs = [(T, N) for N in Ns for T in Ts]
    F = np.array(_mapa_paralelo(_run5, jobs, procs, "Figura L5")).reshape(len(Ns), len(Ts))
    nbs = np.array([nbar_of(P['wb'], T) for T in Ts])
    _salvar('figL5.npz', Ts=Ts, Ns=np.array(Ns), Fopt=F, nbs=nbs, kappas=kappa(P['g'], nbs),
            wb=P['wb'], g=P['g'], sig=P['sig'])


# ---------- Figura L6: a relacao unificada (exato x formula) e a validacao do modelo estatico ----------
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


def calc6(procs):
    P = L6; wbs = P['wbs']; V = P['val']
    jobs = [(wb, nb) for nb in P['nbs'] for wb in wbs]
    E = np.array(_mapa_paralelo(_run6a, jobs, procs, "Figura L6a")).reshape(len(P['nbs']), len(wbs))
    jobs = [(N, s) for s in V['sigs'] for N in V['Ns']]
    Q = np.array(_mapa_paralelo(_run6b, jobs, procs, "Figura L6b")).reshape(len(V['sigs']), len(V['Ns']), 4)
    _salvar('figL6.npz', wbs=wbs, nbs=np.array(P['nbs']), g=P['g'], perda=E,
            val_Ns=np.array(V['Ns']), val_sigs=np.array(V['sigs']), val_wb=V['wb'], val_g=V['g'],
            F_exato=Q[:, :, 0], F_estatico=Q[:, :, 1], F_CC=Q[:, :, 2], absC_exato=Q[:, :, 3])


CALCS = {1: calc1, 2: calc2, 3: calc3, 4: calc4, 5: calc5, 6: calc6}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Calculo das figuras do regime lento.")
    ap.add_argument('--figs', type=int, nargs='+', default=list(CALCS), choices=list(CALCS))
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    print(f"dados em: {os.path.join(os.getcwd(), PASTA)}")
    t0 = time.time()
    for k in a.figs:
        CALCS[k](a.procs)
    print(f"Total: {_fmt_t(time.time() - t0)}")
