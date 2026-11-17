"""Calculo das figuras do qubit dual-rail de 1 foton em N sitios com osciladores termicos.
Salva os resultados em ./data/figN{k}.npz (diretorio corrente) para plotar.py.
Requer nsitios.py e core.py na mesma pasta.

Uso:
    python calcular.py                # todas as figuras
    python calcular.py --figs 2 5     # so algumas
    python calcular.py --procs 4
Mostra uma barra por figura (pontos, %, decorrido, restante) e uma barra global
com o tempo total restante estimado.
"""
import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsitios import cadeia, cadeia_ressonante  # noqa: E402

PASTA = 'data'   # relativa ao diretorio corrente


def nbar_of(wb, T):
    """Ocupacao termica media do oscilador."""
    return 0.0 if T == 0 else 1.0 / np.expm1(wb / T)


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


def _msg(texto):
    (_G.barra.write if _G.barra is not None else print)(texto)


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
        _msg(f"{desc}: {total} pontos em {_fmt_t(time.time() - t0)}")
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


def _salvar(nome, **dados):
    os.makedirs(PASTA, exist_ok=True)
    caminho = os.path.join(os.getcwd(), PASTA, nome)
    np.savez(caminho, **dados)
    _msg(f"salvo: {caminho}")
    return caminho


# =====================================================================
#  Parametros de cada figura
# =====================================================================
N1 = dict(wb=10.0, g=0.5, sig=0.1, T=5.0, Ns=[1, 2, 5, 10])
N2 = dict(wb=10.0, sig=0.1, T=5.0, gs=[0.158, 0.316, 0.5, 1.0], Ns=list(range(1, 31)))
N3 = dict(sig=0.1, T=1.0, N=10, wbs=np.geomspace(0.5, 30, 14), gs=np.geomspace(0.02, 1.5, 14))
N4 = dict(wb=10.0, g=0.5, T=5.0, Ns=[1, 5, 10], sigs=np.geomspace(0.005, 2, 18))
N5 = dict(wb=10.0, g=0.3, sig=0.1, Ns=[1, 5, 10, 20], Ts=np.linspace(0.0, 30.0, 25))


def _nus1():
    return np.unique(np.concatenate([np.linspace(-13, -2, 120), np.linspace(-2, 2, 241),
                                     np.linspace(2, 13, 120)]))


# ---------- Figura N1: varredura em frequencia ----------
def _run1(a):
    nu, N = a
    P = N1; nb = nbar_of(P['wb'], P['T']); Dref = -4*P['g']**2/P['wb']
    return cadeia(nu, P['sig'], N, P['wb'], P['g'], nb, Dref=Dref)


def calc1(procs):
    P = N1; wb, g, Ns = P['wb'], P['g'], P['Ns']
    nus = _nus1()
    jobs = [(nu, N) for N in Ns for nu in nus]
    R = _mapa_paralelo(_run1, jobs, procs, "Figura N1")

    def campo(k):
        return np.array([q[k] for q in R]).reshape(len(Ns), len(nus))

    _salvar('figN1.npz', nus=nus, Ns=np.array(Ns),
            eta=campo('eta'), C=campo('C'), Fbar=campo('Fbar'), Fopt=campo('Fopt'),
            phase=campo('phase'),
            wb=wb, g=g, sig=P['sig'], T=P['T'], nb=nbar_of(wb, P['T']),
            S=(2*g/wb)**2, Dref=-4*g**2/wb)


# ---------- Figura N2: acumulo com N ----------
def _run2(a):
    N, g = a
    P = N2
    return cadeia_ressonante(P['sig'], N, P['wb'], g, nbar_of(P['wb'], P['T']))


def calc2(procs):
    P = N2; gs, Ns = P['gs'], np.array(P['Ns'])
    jobs = [(N, g) for g in gs for N in Ns]
    R = _mapa_paralelo(_run2, jobs, procs, "Figura N2")
    sh = (len(gs), len(Ns))
    _salvar('figN2.npz', Ns=Ns, gs=np.array(gs),
            Fopt=np.array([q['Fopt'] for q in R]).reshape(sh),
            eta=np.array([q['eta'] for q in R]).reshape(sh),
            absC=np.array([abs(q['C']) for q in R]).reshape(sh),
            wb=P['wb'], sig=P['sig'], T=P['T'], nb=nbar_of(P['wb'], P['T']))


# ---------- Figura N3: mapa (omega_b, g) para N=10 ----------
def _run3(a):
    wb, g = a
    P = N3
    return cadeia_ressonante(P['sig'], P['N'], wb, g, nbar_of(wb, P['T']), L=81)['Fopt']


def calc3(procs):
    P = N3; wbs, gs = P['wbs'], P['gs']
    jobs = [(wb, g) for g in gs for wb in wbs]
    F = np.array(_mapa_paralelo(_run3, jobs, procs, "Figura N3")).reshape(len(gs), len(wbs))
    _salvar('figN3.npz', wbs=wbs, gs=gs, F=F, sig=P['sig'], T=P['T'], N=P['N'])


# ---------- Figura N4: largura do pacote ----------
def _run4(a):
    sig, N = a
    P = N4
    r = cadeia_ressonante(sig, N, P['wb'], P['g'], nbar_of(P['wb'], P['T']))
    return (r['Fopt'], r['eta'], abs(r['C']))


def calc4(procs):
    P = N4; sigs, Ns = P['sigs'], P['Ns']
    jobs = [(s, N) for N in Ns for s in sigs]
    R = np.array(_mapa_paralelo(_run4, jobs, procs, "Figura N4")).reshape(len(Ns), len(sigs), 3)
    _salvar('figN4.npz', sigs=sigs, Ns=np.array(Ns),
            Fopt=R[:, :, 0], eta=R[:, :, 1], absC=R[:, :, 2],
            wb=P['wb'], g=P['g'], T=P['T'], nb=nbar_of(P['wb'], P['T']), S=(2*P['g']/P['wb'])**2)


# ---------- Figura N5: temperatura ----------
def _run5(a):
    T, N = a
    P = N5
    return cadeia_ressonante(P['sig'], N, P['wb'], P['g'], nbar_of(P['wb'], T))['Fopt']


def calc5(procs):
    P = N5; Ts, Ns = P['Ts'], P['Ns']
    jobs = [(T, N) for N in Ns for T in Ts]
    F = np.array(_mapa_paralelo(_run5, jobs, procs, "Figura N5")).reshape(len(Ns), len(Ts))
    _salvar('figN5.npz', Ts=Ts, Ns=np.array(Ns), Fopt=F,
            nbs=np.array([nbar_of(P['wb'], T) for T in Ts]),
            wb=P['wb'], g=P['g'], sig=P['sig'], S=(2*P['g']/P['wb'])**2)


def _n_pontos(k):
    """Numero de pontos calculados em cada figura (para a barra global)."""
    return {1: len(N1['Ns']) * len(_nus1()),
            2: len(N2['gs']) * len(N2['Ns']),
            3: len(N3['gs']) * len(N3['wbs']),
            4: len(N4['Ns']) * len(N4['sigs']),
            5: len(N5['Ns']) * len(N5['Ts'])}[k]


CALCS = {1: calc1, 2: calc2, 3: calc3, 4: calc4, 5: calc5}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Calculo: 1 foton em N sitios com osciladores termicos.")
    ap.add_argument('--figs', type=int, nargs='+', default=list(CALCS), choices=list(CALCS))
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    print(f"dados em: {os.path.join(os.getcwd(), PASTA)}")
    t_total = time.time()
    _iniciar_global(sum(_n_pontos(k) for k in a.figs))
    try:
        for i, k in enumerate(a.figs, 1):
            _msg(f"[{i}/{len(a.figs)}] Figura N{k}: {_n_pontos(k)} pontos")
            t0 = time.time()
            CALCS[k](a.procs)
            _msg(f"[{i}/{len(a.figs)}] Figura N{k} concluida em {_fmt_t(time.time() - t0)}")
    finally:
        _fechar_global()
    print(f"Total: {_fmt_t(time.time() - t_total)}")
