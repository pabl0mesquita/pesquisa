"""Qubit dual-rail de 1 foton atravessando N sitios de Brod-Combes em cascata.

Cada sitio j: atomo A_j acoplado ao guia (trilho 1) e a SEU PROPRIO oscilador b_j por
g(1 - A_z^{(j)})(b_j + b_j^dagger), com b_j inicialmente termico (independentes, identicos).
Trilho 2 livre. Ideal: N atomos sem oscilador, t_0(omega)^N (descontando a deformacao
de foton unico, como Brod-Combes).

Para 1 foton num guia unidirecional, a matriz-S da cadeia e a composicao das matrizes-S
de cada sitio (o foton que sai do sitio j e um foton livre que entra no sitio j+1).
Tracando cada oscilador logo apos seu sitio (os sitios seguintes nao o tocam), o trilho 1
evolui por um canal em cada sitio:

    rho'(w, w') = sum_n p_n sum_m t_mn(nu) t_mn(nu')^* rho(nu, nu'),  nu = w + (m-n) w_b.

O deslocamento (m-n) w_b e o mesmo para w e w', entao rho fica em BLOCOS de banda lateral k
(frequencias  w_c - k w_b + x), e o canal age elemento a elemento em cada bloco.

    eta = <u| rho_final |u>,   u = xi t_0^N
    C   = int dw |xi|^2 [t_0^* tbar]^N,   tbar = sum_n p_n t_nn      (fechado e exato)
    Fbar = (1 + eta + Re C)/3

Requer core.py na mesma pasta.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import t_mn, t0, xi, thermal_p, disp_matrix, line_center, nbar_of  # noqa: E402,F401


def _setup(wb, g, nbar, fock=None, nmax=None, side=None):
    if fock is not None:
        nmax = int(fock) if nmax is None else nmax
        p = np.zeros(nmax + 1); p[int(fock)] = 1.0
        nb_eff = float(fock)
    else:
        nmax = int(max(6, nbar * 12 + 6)) if nmax is None else nmax
        p = thermal_p(nbar, nmax); p = p / p.sum()
        nb_eff = nbar
    if side is None:
        side = int(min(30, 4 + 6 * (2 * g / wb) ** 2 * (2 * nb_eff + 1) + 3))
    N = nmax + side + 60
    D = disp_matrix(2 * g / wb, N)
    return p, nmax, side, D, N - 20


def cadeia(wc, sig, N, wb, g, nbar, Dref, Delta=0.0, L=121, fock=None, kmax=None, tol=1e-13):
    """Retorna dict com eta, C, Fbar, Fopt, unit (soma dos tracos), fase ideal."""
    p, nmax, side, Dm, Kfc = _setup(wb, g, nbar, fock)
    W = 6 * sig
    x = np.linspace(-W, W, L); dx = x[1] - x[0]
    if kmax is None:
        kmax = 3 * side
    v = xi(wc + x, wc, sig) * np.sqrt(dx)
    blocos = {0: np.outer(v, v.conj())}
    ns = [n for n in range(nmax + 1) if p[n] > 1e-12]
    cacheT = {}

    def kernel(k, d):
        chave = (k, d)
        if chave not in cacheT:
            nu = wc - k * wb + x
            linhas, pesos = [], []
            for n in ns:
                m = n + d
                if m < 0:
                    continue
                linhas.append(t_mn(m, n, nu, Delta, wb, g, Dm, Kfc)); pesos.append(p[n])
            if not linhas:
                cacheT[chave] = None
            else:
                T = np.array(linhas); w = np.array(pesos)
                cacheT[chave] = (T * w[:, None]).T @ T.conj()     # K[x,x'] = sum_n p_n t(x) t*(x')
        return cacheT[chave]

    for _ in range(N):
        novos = {}
        for k, R in blocos.items():
            if abs(np.trace(R)) < tol:
                continue
            for d in range(-side, side + 1):
                kk = k + d
                if abs(kk) > kmax:
                    continue
                K = kernel(k, d)
                if K is None:
                    continue
                novos[kk] = novos.get(kk, 0) + K * R
        blocos = novos

    unit = sum(np.trace(R).real for R in blocos.values())
    eta = 0.0
    for k, R in blocos.items():
        nu = wc - k * wb + x
        w = xi(nu, wc, sig) * t0(nu, Dref) ** N * np.sqrt(dx)
        eta += (w.conj() @ R @ w).real
    # coerencia: fechada
    xg = np.linspace(-W, W, 4 * L + 1); dxg = xg[1] - xg[0]; nu = wc + xg
    tbar = sum(p[n] * t_mn(n, n, nu, Delta, wb, g, Dm, Kfc) for n in ns)
    C = np.sum(abs(xi(nu, wc, sig)) ** 2 * (np.conj(t0(nu, Dref)) * tbar) ** N) * dxg
    fase = np.angle(np.sum(abs(xi(nu, wc, sig)) ** 2 * t0(nu, Dref) ** N) * dxg)
    return dict(eta=eta, C=C, Fbar=(1 + eta + C.real) / 3, Fopt=(1 + eta + abs(C)) / 3,
                unit=unit, phase=fase)


def cadeia_ressonante(sig, N, wb, g, nbar, L=121):
    c = line_center(wb, g, nbar)
    r = cadeia(c, sig, N, wb, g, nbar, Dref=c, L=L); r['wc'] = c
    return r