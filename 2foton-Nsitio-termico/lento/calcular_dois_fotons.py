"""Calculo das figuras da porta CZ de Brod-Combes (contrapropagante, chi -> inf) com
osciladores no regime lento. Salva ./data/dois_fotons_dados.npz (diretorio corrente)
para plotar_dois_fotons.py. Requer dois_fotons.py na mesma pasta.

Calcula:
    R    F(sigma) para cada (kappa, N)                         -> figs. D1, D2, D3
    R2   F em grade fina ao redor do sigma otimo, kgrid        -> fig. D4
    D    dF/dkappa em kappa=0 (diferencas finitas)             -> figs. D4, D5

Uso:
    python calcular_dois_fotons.py --procs 4
    python calcular_dois_fotons.py --forcar     # recalcula mesmo se o .npz ja existir
"""
import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Pool
from scipy.stats import qmc, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dois_fotons import mus, F_de_mus, gam  # noqa: E402

PASTA = 'data'                     # relativa ao diretorio corrente
ARQUIVO = 'dois_fotons_dados.npz'

# =====================================================================
#  Parametros
# =====================================================================
NS = [1, 3, 5]
SIGS = np.logspace(-2.5, np.log10(0.6), 24)
KAPS = [0, 1e-3, 1e-2, 5e-2]
KGRID = np.array([0, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1])   # kappas da fig. D4
N_FINO = 7                         # pontos da grade fina em sigma ao redor do otimo (fig. D4)
M = 128                            # pontos de Sobol por media
L = 48                             # pontos da grade de frequencias
H = 0.05                           # passo das diferencas finitas em dF/dkappa


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


def dFdk(N, sig, h=H):
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


# =====================================================================
#  Progresso
# =====================================================================
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


# =====================================================================
#  Calculo
# =====================================================================
def calcular(procs):
    jobs = [(N, s, k) for k in KAPS for N in NS for s in SIGS]
    with Pool(procs) as p:
        R = np.array(_barra(p, _job, jobs, '[1/3] F(sigma)')).reshape(len(KAPS), len(NS), len(SIGS))
        # F_max: maximiza em grade fina ao redor do otimo de kappa=0
        sfin = np.array([SIGS[np.argmax(R[0, i])] * np.logspace(-0.25, 0.25, N_FINO)
                         for i in range(len(NS))])
        jobs2 = [(N, s, k) for k in KGRID for i, N in enumerate(NS) for s in sfin[i]]
        R2 = np.array(_barra(p, _job, jobs2, '[2/3] F_max(kappa)')).reshape(len(KGRID), len(NS), N_FINO)
        jobs3 = [(N, s) for N in NS for s in SIGS]
        D = np.array(_barra(p, _jobd, jobs3, '[3/3] dF/dkappa')).reshape(len(NS), len(SIGS))
    return dict(R=R, kgrid=KGRID, R2=R2, sfin=sfin, D=D,
                NS=np.array(NS), SIGS=SIGS, KAPS=np.array(KAPS, float), M=M, L=L, h=H, gam=gam)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Calculo: CZ contrapropagante com osciladores lentos.")
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--forcar', action='store_true', help="recalcula mesmo se o .npz ja existir")
    A = ap.parse_args()
    caminho = os.path.join(os.getcwd(), PASTA, ARQUIVO)
    if os.path.exists(caminho) and not A.forcar:
        print(f"{caminho} ja existe; use --forcar para recalcular.")
        sys.exit(0)
    dados = calcular(A.procs)
    os.makedirs(PASTA, exist_ok=True)
    np.savez(caminho, **dados)
    print(f"salvo: {caminho}")
    print(f"ok -- tempo total {_hms(time.time() - T_INICIO)}")
